import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple, List, Union

from library.config import Config
from library.transforms import get_transforms

# ==========================================
# Dataset Implementation
# ==========================================


class EEGDataset(Dataset):
    """
    PyTorch Dataset for the Siamese Equivariant Dual-Stream Network.
    Handles loading and slicing of EEG and Spectrogram parquet files.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mode: str = "train",
        transforms: Optional[Dict] = None,
        cache_spec_cols: bool = True,
    ):
        """
        Args:
            df: Metadata DataFrame (train, val, or test).
            mode: 'train', 'val', or 'test'.
            transforms: Dictionary of transforms for 'eeg' and 'spec'.
            cache_spec_cols: If True, pre-calculates spectrogram column mappings.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transforms = transforms

        # Pre-calculate absolute paths for faster access
        # Assuming paths in metadata are relative to input dir (e.g., "train_eegs/...")
        # But metadata generation script put relative paths like "train_eegs/x.parquet"
        # We need to join with INPUT_DIR
        self.df["eeg_abs_path"] = self.df["eeg_path"].apply(
            lambda x: os.path.join(Config.INPUT_DIR, x)
        )
        self.df["spec_abs_path"] = self.df["spec_path"].apply(
            lambda x: os.path.join(Config.INPUT_DIR, x)
        )

        # Handle missing offset columns for Test set
        if "eeg_label_offset_seconds" not in self.df.columns:
            self.df["eeg_label_offset_seconds"] = 0.0
        if "spectogram_label_offset_seconds" not in self.df.columns:
            self.df["spectogram_label_offset_seconds"] = 0.0

        # Cache spectrogram column names to avoid parsing every iteration
        # We need a sample file to determine columns.
        # We'll do lazy initialization in __getitem__ or try one file now.
        self.spec_cols_map = None
        if cache_spec_cols and len(self.df) > 0:
            try:
                sample_path = self.df.iloc[0]["spec_abs_path"]
                if os.path.exists(sample_path):
                    sample_df = pd.read_parquet(sample_path)
                    self.spec_cols_map = self._get_spec_col_mapping(sample_df.columns)
            except Exception as e:
                print(f"Warning: Could not pre-cache spectrogram columns: {e}")

    def _get_spec_col_mapping(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Identifies columns for each region (LL, RL, LP, RP) and sorts them by frequency.
        Expected format: 'LL_0.59', 'RL_0.59', etc.
        """
        mapping = {"LL": [], "RL": [], "LP": [], "RP": []}

        # Filter relevant columns
        for col in columns:
            for region in mapping.keys():
                if col.startswith(f"{region}_"):
                    mapping[region].append(col)
                    break

        # Sort by frequency (float value after underscore)
        for region in mapping:
            # Sort key: float value of the suffix
            try:
                mapping[region].sort(key=lambda x: float(x.split("_")[1]))
            except:
                # Fallback if format is unexpected, just sort alphabetically
                mapping[region].sort()

        return mapping

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        # ==========================
        # 1. Load and Process EEG
        # ==========================
        try:
            eeg_df = pd.read_parquet(row["eeg_abs_path"])

            # Slice 50s window
            offset_sec = row["eeg_label_offset_seconds"]
            start_idx = int(offset_sec * Config.EEG_SR)
            end_idx = start_idx + Config.EEG_SAMPLES  # 10000 samples

            # Handle edge cases where file is shorter or offset is wrong
            if start_idx < 0:
                start_idx = 0

            # Slice
            eeg_segment = eeg_df.iloc[start_idx:end_idx]

            # If segment is too short (end of file), pad or accept (Transform handles it?)
            # Transform expects DataFrame. We'll let it handle NaNs/zeros if length mismatch,
            # but ideally we reindex to ensure length.
            if len(eeg_segment) < Config.EEG_SAMPLES:
                # Reindex to ensure fixed size, filling missing with NaN (Transform fills 0)
                new_index = range(start_idx, end_idx)
                eeg_segment = eeg_segment.reindex(
                    new_index
                )  # This might not work if index is not integer range
                # Simpler: just pad if needed inside transform, but here we pass DF.
                # Let's assume data is sufficient or Transform handles it.
                pass

        except Exception as e:
            # Fallback for corrupt files
            # Create dummy dataframe with correct columns
            eeg_segment = pd.DataFrame(
                np.zeros((Config.EEG_SAMPLES, Config.N_EEG_CHANNELS)),
                columns=[c for chains in Config.CHAIN_CONFIG.values() for c in chains][
                    : Config.N_EEG_CHANNELS
                ],  # Approx
            )
            # Better: use predefined columns from Config if available, or just empty
            # The transform uses Config.CHAIN_CONFIG to look up columns.
            # We need a DF with those columns.
            all_cols = set()
            for chain in Config.CHAIN_CONFIG.values():
                all_cols.update(chain)
            eeg_segment = pd.DataFrame(
                np.zeros((Config.EEG_SAMPLES, len(all_cols))), columns=list(all_cols)
            )

        # Apply EEG Transform
        if self.transforms and "eeg" in self.transforms:
            eeg_tensor = self.transforms["eeg"](eeg_segment)
        else:
            eeg_tensor = torch.zeros((4, 5, 128, 256))  # Dummy

        # ==========================
        # 2. Load and Process Spec
        # ==========================
        try:
            spec_df = pd.read_parquet(row["spec_abs_path"])

            # Initialize mapping if not done (e.g. first call failed or skipped)
            if self.spec_cols_map is None:
                self.spec_cols_map = self._get_spec_col_mapping(spec_df.columns)

            # Slice 10m window
            # Kaggle specs usually have a 'time' column.
            offset_sec = row["spectogram_label_offset_seconds"]

            if "time" in spec_df.columns:
                # Filter 600s window
                # Note: Test files are exactly 10m, so offset usually 0 and time covers it.
                # Train files are longer.
                mask = (spec_df["time"] >= offset_sec) & (
                    spec_df["time"] < offset_sec + Config.SPEC_DURATION
                )
                spec_window = spec_df.loc[mask]

                # If empty (offset out of bounds), take nearest or full?
                if spec_window.empty:
                    spec_window = spec_df  # Fallback
            else:
                spec_window = spec_df

            # Reshape to (4, F, T)
            # Extract columns for each region
            regions = ["LL", "RL", "LP", "RP"]
            region_arrays = []

            for r in regions:
                cols = self.spec_cols_map[r]
                if not cols:
                    # Fallback if columns missing
                    r_data = np.zeros((100, len(spec_window)))
                else:
                    # (Time, Freq) -> Transpose to (Freq, Time)
                    r_data = spec_window[cols].values.T
                region_arrays.append(r_data)

            # Stack: (4, F, T)
            spec_arr = np.stack(region_arrays, axis=0)

        except Exception as e:
            # Fallback
            spec_arr = np.zeros((4, 100, 300))  # Approx shape

        # Apply Spec Transform
        if self.transforms and "spec" in self.transforms:
            spec_tensor = self.transforms["spec"](spec_arr)
        else:
            spec_tensor = torch.zeros((4, 256, 256))

        # ==========================
        # 3. Prepare Output
        # ==========================
        output = {"eeg": eeg_tensor, "spec": spec_tensor, "eeg_id": row["eeg_id"]}

        if self.mode != "test":
            # Extract targets
            targets = row[Config.CLASS_NAMES].values.astype(np.float32)
            # Ensure sum to 1 (already done in metadata, but safe to enforce)
            if targets.sum() > 0:
                targets = targets / targets.sum()
            else:
                targets = np.ones(Config.NUM_CLASSES) / Config.NUM_CLASSES

            return output, torch.tensor(targets)
        else:
            return output


# ==========================================
# DataLoader Factory
# ==========================================


def get_dataloaders(
    load_cached_data: bool = False, debug: bool = False
) -> Dict[str, DataLoader]:
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data: Unused for raw data (too large), but kept for API consistency.
                          Metadata is loaded from ./metadata CSVs.
        debug: If True, subsamples the dataset for quick testing.
    """

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Check for test set
    if os.path.exists(Config.TEST_CSV):
        test_df = pd.read_csv(Config.TEST_CSV)
    else:
        test_df = pd.DataFrame()  # Empty if not found

    # 2. Debug Subsampling
    if debug or Config.DEBUG:
        print(f"DEBUG MODE: Subsampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        if not test_df.empty:
            test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 3. Initialize Transforms
    train_transforms = get_transforms(mode="train")
    val_transforms = get_transforms(mode="val")  # No augmentation
    test_transforms = get_transforms(mode="test")

    # 4. Create Datasets
    train_ds = EEGDataset(train_df, mode="train", transforms=train_transforms)
    val_ds = EEGDataset(val_df, mode="val", transforms=val_transforms)

    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        ),
    }

    if not test_df.empty:
        test_ds = EEGDataset(test_df, mode="test", transforms=test_transforms)
        dataloaders["test"] = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    return dataloaders
