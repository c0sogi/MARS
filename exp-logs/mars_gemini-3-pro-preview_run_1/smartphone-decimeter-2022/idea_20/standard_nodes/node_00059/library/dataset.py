import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import CFG
from library.data_preprocessing import process_dataset


class GnssSequenceDataset(Dataset):
    def __init__(
        self, df, feature_cols, target_cols=None, mode="train", sequence_length=128
    ):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.mode = mode
        self.sequence_length = sequence_length

        # Ensure indices are accessible for mapping back later
        self.df["orig_index"] = self.df.index

        # Group data by drive and phone to maintain sequence integrity
        self.groups = [g for _, g in self.df.groupby(["drive_id", "phone_name"])]

        self.samples = []

        # Define stride: overlap for training, non-overlap for evaluation
        if mode == "train":
            stride = sequence_length // 2
        else:
            stride = sequence_length

        # Create windows
        for group in self.groups:
            num_rows = len(group)
            # Sort by time just in case
            group = group.sort_values("UnixTimeMillis")

            for start_idx in range(0, num_rows, stride):
                end_idx = min(start_idx + sequence_length, num_rows)

                # For inference/val, we want to cover everything.
                # If the last chunk is smaller than sequence_length, we still take it and pad later.
                # For train, we can drop very short sequences if we wanted, but padding is safer.

                # Store the slice of the dataframe corresponding to this window
                self.samples.append(group.iloc[start_idx:end_idx])

                # If we've reached the end, break
                if end_idx == num_rows:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_df = self.samples[idx]
        seq_len = len(sample_df)

        # Extract features
        features = sample_df[self.feature_cols].values.astype(np.float32)

        # Extract targets if available
        if self.target_cols:
            targets = sample_df[self.target_cols].values.astype(np.float32)
        else:
            # Dummy targets for test set
            targets = np.zeros((seq_len, 2), dtype=np.float32)

        # Extract indices
        indices = sample_df["orig_index"].values.astype(np.int64)

        # Padding
        if seq_len < self.sequence_length:
            pad_size = self.sequence_length - seq_len

            # Pad features with 0
            features = np.pad(
                features, ((0, pad_size), (0, 0)), mode="constant", constant_values=0
            )

            # Pad targets with 0
            targets = np.pad(
                targets, ((0, pad_size), (0, 0)), mode="constant", constant_values=0
            )

            # Pad indices with -1
            indices = np.pad(
                indices, (0, pad_size), mode="constant", constant_values=-1
            )

            # Mask: 1 for valid data, 0 for padding
            mask = np.concatenate(
                [
                    np.ones(seq_len, dtype=np.float32),
                    np.zeros(pad_size, dtype=np.float32),
                ]
            )
        else:
            mask = np.ones(seq_len, dtype=np.float32)

        # Transpose inputs to (Channels, Length) for 1D CNN
        # features: (L, C) -> (C, L)
        features = torch.tensor(features).permute(1, 0)

        # targets: (L, 2) -> (2, L)
        targets = torch.tensor(targets).permute(1, 0)

        mask = torch.tensor(mask)
        indices = torch.tensor(indices)

        return features, targets, mask, indices


def get_scaler(df, feature_cols):
    """Fits a StandardScaler on the provided DataFrame's feature columns."""
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    return scaler


def get_train_val_datasets(load_cached_data=True, debug=False):
    """
    Loads train and validation data, scales features, and returns Dataset objects.
    """
    # 1. Load Data
    df_train = process_dataset(
        CFG.TRAIN_METADATA_PATH,
        CFG.TRAIN_CACHE_PATH,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    df_val = process_dataset(
        CFG.VAL_METADATA_PATH,
        CFG.VAL_CACHE_PATH,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # 2. Fit Scaler on Training Data Only
    print("Fitting scaler on training data...")
    scaler = get_scaler(df_train, CFG.FEATURE_COLS)

    # 3. Transform Data
    print("Transforming features...")
    df_train[CFG.FEATURE_COLS] = scaler.transform(df_train[CFG.FEATURE_COLS])
    df_val[CFG.FEATURE_COLS] = scaler.transform(df_val[CFG.FEATURE_COLS])

    # 4. Create Datasets
    # Targets are 'target_east' and 'target_north' generated in preprocessing
    target_cols = ["target_east", "target_north"]

    train_dataset = GnssSequenceDataset(
        df_train,
        CFG.FEATURE_COLS,
        target_cols,
        mode="train",
        sequence_length=CFG.SEQUENCE_LENGTH,
    )

    val_dataset = GnssSequenceDataset(
        df_val,
        CFG.FEATURE_COLS,
        target_cols,
        mode="val",
        sequence_length=CFG.SEQUENCE_LENGTH,
    )

    return train_dataset, val_dataset, scaler


def get_test_dataset(scaler, load_cached_data=True, debug=False):
    """
    Loads test data, scales features using the provided scaler, and returns Dataset object.
    """
    # 1. Load Data
    df_test = process_dataset(
        CFG.TEST_METADATA_PATH,
        CFG.TEST_CACHE_PATH,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # 2. Transform Data
    print("Transforming test features...")
    # Handle NaNs in test if any (though preprocessing should handle it)
    df_test[CFG.FEATURE_COLS] = df_test[CFG.FEATURE_COLS].fillna(0)
    df_test[CFG.FEATURE_COLS] = scaler.transform(df_test[CFG.FEATURE_COLS])

    # 3. Create Dataset
    test_dataset = GnssSequenceDataset(
        df_test,
        CFG.FEATURE_COLS,
        target_cols=None,  # No targets for test
        mode="test",
        sequence_length=CFG.SEQUENCE_LENGTH,
    )

    return test_dataset
