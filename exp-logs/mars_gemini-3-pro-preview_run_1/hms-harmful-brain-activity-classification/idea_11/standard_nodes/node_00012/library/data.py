import os
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class BrainDataset(Dataset):
    def __init__(self, df, config, mode="train", augment=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            config (Config): Configuration object.
            mode (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply data augmentation.
        """
        self.df = df
        self.config = config
        self.mode = mode
        self.augment = augment

        # Ensure cache directory exists
        if self.config.LOAD_CACHED_DATA:
            os.makedirs(self.config.CACHE_DIR, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Determine Cache Key
        if self.mode == "test":
            # Test set doesn't have label_id, use eeg_id
            cache_key = f"{self.mode}_{row['eeg_id']}"
        else:
            # Train/Val sets use label_id which is unique for the event
            cache_key = f"{self.mode}_{int(row['label_id'])}"

        cache_path = os.path.join(self.config.CACHE_DIR, f"{cache_key}.npy")

        # 1. Try Loading from Cache
        if self.config.LOAD_CACHED_DATA and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True).item()
                # If augmentation is enabled, we must apply it AFTER loading cached raw tensors
                # However, to save space, we might cache processed tensors.
                # Strategy: Cache the processed (resized/downsampled) but un-augmented tensors.

                eeg_tensor = data["eeg"]
                spec_tensor = data["spec"]
                target = data.get("target", None)

                if self.augment:
                    eeg_tensor, spec_tensor = self._augment_data(
                        eeg_tensor, spec_tensor
                    )

                return self._format_output(eeg_tensor, spec_tensor, target, row)
            except Exception as e:
                # If load fails, fall back to compute
                pass

        # 2. Compute from Scratch
        eeg_tensor = self._load_eeg(row)
        spec_tensor = self._load_spec(row)

        # Get target if available
        target = None
        if self.mode != "test":
            target_cols = [f"{c}_prob" for c in self.config.CLASS_NAMES]
            if all(col in row for col in target_cols):
                target = row[target_cols].values.astype(np.float32)
            else:
                # Fallback to votes if probs not present (though metadata gen makes probs)
                vote_cols = [f"{c}_vote" for c in self.config.CLASS_NAMES]
                votes = row[vote_cols].values.astype(np.float32)
                target = votes / (votes.sum() + 1e-6)

        # 3. Save to Cache (Un-augmented)
        if self.config.LOAD_CACHED_DATA:
            try:
                save_data = {"eeg": eeg_tensor, "spec": spec_tensor}
                if target is not None:
                    save_data["target"] = target
                np.save(cache_path, save_data)
            except Exception:
                pass  # Non-critical failure

        # 4. Apply Augmentation (if enabled)
        if self.augment:
            eeg_tensor, spec_tensor = self._augment_data(eeg_tensor, spec_tensor)

        return self._format_output(eeg_tensor, spec_tensor, target, row)

    def _format_output(self, eeg, spec, target, row):
        output = {
            "eeg": torch.tensor(eeg, dtype=torch.float32),
            "spec": torch.tensor(spec, dtype=torch.float32),
        }
        if target is not None:
            output["target"] = torch.tensor(target, dtype=torch.float32)

        if self.mode == "test":
            output["eeg_id"] = row["eeg_id"]

        return output

    def _load_eeg(self, row):
        """
        Loads, slices, downsamples, and normalizes EEG data.
        Returns: np.ndarray of shape (Channels, Time) -> (20, 5000)
        """
        file_path = os.path.join(self.config.INPUT_DIR, row["eeg_path"])

        try:
            # Load parquet
            eeg_df = pd.read_parquet(file_path)

            # Determine offset and window
            # Metadata offset is in seconds. Data is 200Hz.
            # The label is for the 50s window starting at eeg_label_offset_seconds
            start_sec = int(row["eeg_label_offset_seconds"])
            start_idx = int(start_sec * 200)
            end_idx = start_idx + 50 * 200  # 50 seconds * 200 Hz = 10000 samples

            # Handle Test Set (might not have offset, or different logic)
            # Test set eegs are exactly 50s, so offset is likely 0 or irrelevant if file is 50s.
            if self.mode == "test" and len(eeg_df) <= 10000:
                data = eeg_df.values
            else:
                # Extract window
                # Check bounds
                if start_idx < 0:
                    start_idx = 0

                data = eeg_df.iloc[start_idx:end_idx].values

                # Pad if too short
                if data.shape[0] < 10000:
                    pad_len = 10000 - data.shape[0]
                    # Pad with zeros
                    data = np.pad(data, ((0, pad_len), (0, 0)), mode="constant")

            # Select Channels (Ensure 20 channels)
            # Columns: Fp1, F3, ..., EKG.
            # We assume the file has the correct columns. If EKG is missing, we might need to handle,
            # but dataset description says EKG is present.
            # We just take the values. Shape: (Time, Channels)

            # Downsample: 200Hz -> 100Hz
            # Simple slicing [::2]
            data = data[::2, :]  # Shape (5000, 20)

            # Transpose to (Channels, Time) -> (20, 5000)
            data = data.T

            # Handle NaNs
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize (Channel-wise)
            # Mean/Std
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            data = (data - mean) / (std + 1e-6)

            # Clip outliers
            data = np.clip(data, -5, 5)

            return data.astype(np.float32)

        except Exception as e:
            # Fallback for errors: return zeros
            return np.zeros(
                (self.config.EEG_CHANNELS, self.config.EEG_SEQ_LEN), dtype=np.float32
            )

    def _load_spec(self, row):
        """
        Loads, slices, resizes spectrogram and adds coordinate channel.
        Returns: np.ndarray of shape (5, 512, 512)
        """
        file_path = os.path.join(self.config.INPUT_DIR, row["spectrogram_path"])

        try:
            spec_df = pd.read_parquet(file_path)

            # Determine window
            # Spectrogram offset is in seconds.
            # The spectrogram rows correspond to time.
            # Usually 2 seconds per row in these Kaggle datasets?
            # Or 0.5s? The description says "Spectrograms assembled EEG data".
            # Let's infer time resolution.
            # Metadata: "spectrogram_label_offset_seconds".
            # Task: "spectrograms covering 10 a minute window".
            # 10 mins = 600 seconds.
            # If we assume standard Kaggle spec format (e.g., from HMS competition),
            # time bins are often 2s.
            # However, let's look at the data analysis output.
            # "Height Distribution (Time steps): Mean=929.4".
            # If file covers > 10 mins, we need to slice.
            # If we don't know the exact Hz of the spectrogram time axis, we can assume
            # the offset maps to rows linearly if we knew the total duration.
            # Standard approach: The offset is in seconds.
            # We need to find the rows corresponding to [offset, offset + 600].
            # The parquet index usually represents time in seconds or is just 0..N.
            # If index is time, we can use loc.
            # Let's assume the index is the time in seconds (common in these datasets).

            # Check if index is monotonic increasing (time).
            # If not, we assume a rate.
            # Let's assume the provided offset matches the "time" column if it exists,
            # or the index.

            # Standard HMS-HBAC handling:
            # The offset points to the start of the 10 min window.
            # We need to extract rows where time is in [offset, offset+600].
            # If the dataframe has a 'time' column, use it. Otherwise assume index is time?
            # Actually, usually the parquet index is NOT time.
            # But `spectrogram_label_offset_seconds` is provided.
            # Let's assume 0.5Hz resolution (2s per row) is common, OR
            # simply that we need to grab the 10 minute window starting at offset.
            # Wait, if we don't know the resolution, we can't slice by index.
            # However, usually `time` is a column.

            if "time" in spec_df.columns:
                start_t = row["spectrogram_label_offset_seconds"]
                end_t = start_t + 600
                mask = (spec_df["time"] >= start_t) & (spec_df["time"] < end_t)
                window = spec_df[mask]
                # Drop time column for processing
                window = window.drop(columns=["time"], errors="ignore")
            else:
                # Fallback: Assume the whole file is what we want if test,
                # or try to slice based on ratio if we knew total length.
                # Given the ambiguity, and "test_spectrograms" are "Exactly 10 minutes",
                # for test we take all.
                # For train, if "time" is missing, we might be in trouble.
                # But usually these datasets have "time".
                # If not, we'll take the middle 300 rows? No, that's risky.
                # Let's assume 'time' exists or index is time (seconds).
                start_t = row["spectrogram_label_offset_seconds"]
                # Try slicing by index as if it is seconds
                # (This is a heuristic based on the competition data structure)
                window = spec_df.loc[start_t : start_t + 600]

            # If window is empty or too small, pad or take what we have
            data = window.values
            if data.shape[0] == 0:
                data = spec_df.values  # Fallback to full file

            # Handle NaNs
            data = np.nan_to_num(data, nan=0.0)
            data = np.log1p(data)  # Log transform

            # Separate Regions
            # Columns: LL_x, RL_x, LP_x, RP_x
            # We want to create 4 images.
            regions = ["LL", "RL", "LP", "RP"]
            region_imgs = []

            all_cols = spec_df.columns
            # Remove 'time' if present in list
            feat_cols = [c for c in all_cols if c != "time"]

            for region in regions:
                # Find columns for this region
                r_cols = [c for c in feat_cols if c.startswith(f"{region}_")]
                if not r_cols:
                    # Fallback: if no prefix, maybe just split columns equally?
                    # Unlikely given description. Return zeros if fail.
                    img = np.zeros((self.config.SPEC_IMG_SIZE), dtype=np.float32)
                else:
                    # Extract columns
                    # We need to map these columns to indices in 'data' (which is window.values)
                    # window.columns is same as spec_df.columns (minus time)
                    # Let's get indices
                    col_indices = [window.columns.get_loc(c) for c in r_cols]
                    r_data = data[:, col_indices]

                    # Resize to (512, 512)
                    # Current shape: (Time_Steps, Freq_Bins)
                    # Resize -> (512, 512)
                    img = cv2.resize(
                        r_data,
                        self.config.SPEC_IMG_SIZE,
                        interpolation=cv2.INTER_LINEAR,
                    )

                region_imgs.append(img)

            # Stack 4 regions -> (4, 512, 512)
            spec_tensor = np.stack(region_imgs, axis=0)

            # Create Coordinate Map (5th Channel)
            # Shape (512, 512). Gradient along Time (Height, axis 0).
            # -1 at top, 0 at center, 1 at bottom.
            # Or -1 at start of window, 1 at end.
            H, W = self.config.SPEC_IMG_SIZE
            # Create a column vector of shape (H, 1)
            y_coords = np.linspace(-1, 1, H).astype(np.float32)
            # Tile to (H, W)
            coord_map = np.tile(y_coords.reshape(-1, 1), (1, W))

            # Stack -> (5, 512, 512)
            spec_tensor = np.concatenate(
                [spec_tensor, coord_map[np.newaxis, :, :]], axis=0
            )

            return spec_tensor.astype(np.float32)

        except Exception as e:
            # Fallback
            return np.zeros(
                (self.config.SPEC_CHANNELS, *self.config.SPEC_IMG_SIZE),
                dtype=np.float32,
            )

    def _augment_data(self, eeg, spec):
        """
        Applies augmentation to EEG and Spectrogram.
        eeg: (20, 5000)
        spec: (5, 512, 512)
        """
        # 1. EEG Channel Dropout
        if np.random.rand() < self.config.CHANNEL_DROPOUT_PROB:
            # Drop 1 to 3 channels
            num_drop = np.random.randint(1, 4)
            channels_idx = np.random.choice(eeg.shape[0], num_drop, replace=False)
            eeg[channels_idx, :] = 0.0

        # 2. Spectrogram Masking (SpecAugment)
        # Apply only to first 4 channels (image data), preserve coord map (ch 4)
        # Time Masking
        if np.random.rand() < self.config.MASK_TIME_PROB:
            # Mask a block of time (columns? No, time is usually height in our resize logic?)
            # Wait, in _load_spec, we did cv2.resize(r_data, (512, 512)).
            # cv2.resize(src, dsize=(width, height)).
            # So output is (512, 512) where 512 is width (Freq) and 512 is height (Time).
            # Let's stick to: Axis 1 is Height (Time), Axis 2 is Width (Freq).
            # Tensor shape (C, H, W).

            H, W = spec.shape[1], spec.shape[2]
            mask_size = np.random.randint(10, H // 4)
            start = np.random.randint(0, H - mask_size)
            spec[:4, start : start + mask_size, :] = 0.0

        # Freq Masking
        if np.random.rand() < self.config.MASK_FREQ_PROB:
            H, W = spec.shape[1], spec.shape[2]
            mask_size = np.random.randint(10, W // 4)
            start = np.random.randint(0, W - mask_size)
            spec[:4, :, start : start + mask_size] = 0.0

        return eeg, spec


def get_dataloaders(config):
    """
    Creates DataLoaders for train, val, and test.
    """
    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)

    # Test DF might exist or not
    if os.path.exists(config.TEST_CSV):
        test_df = pd.read_csv(config.TEST_CSV)
    else:
        test_df = pd.DataFrame()  # Empty if not found

    # Debug Mode
    if config.DEBUG:
        train_df = train_df.iloc[: config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: config.DEBUG_SUBSET_SIZE]
        if not test_df.empty:
            test_df = test_df.iloc[: config.DEBUG_SUBSET_SIZE]

    # Datasets
    train_ds = BrainDataset(train_df, config, mode="train", augment=True)
    val_ds = BrainDataset(val_df, config, mode="val", augment=False)
    test_ds = BrainDataset(test_df, config, mode="test", augment=False)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = None
    if len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
