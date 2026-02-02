import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import WORKING_DIR, METADATA_DIR, BATCH_SIZE, SEED
from library.utils import seed_everything


class ManufacturingDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, target_col="target", is_test=False):
        self.cat_data = df[cat_cols].values.astype(np.int64)
        self.cont_data = df[cont_cols].values.astype(np.float32)
        self.is_test = is_test
        if not is_test:
            self.targets = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        cat = torch.from_numpy(self.cat_data[idx])
        cont = torch.from_numpy(self.cont_data[idx])

        if self.is_test:
            # Return dummy target for test set to maintain consistent signature
            return cat, cont, torch.tensor(0.0)

        target = torch.tensor(self.targets[idx])
        return cat, cont, target


def prepare_data(load_cached_data=True, batch_size=BATCH_SIZE, debug_rows=None):
    """
    Loads, processes, and caches data.
    Implements transductive vocabulary alignment and feature engineering.
    Returns DataLoaders for train, val, and test splits.
    """
    seed_everything(SEED)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    train_cache = os.path.join(WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(WORKING_DIR, "test_processed.parquet")
    meta_cache = os.path.join(WORKING_DIR, "metadata.npy")

    # 1. Check Cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(meta_cache)
    ):
        print("Loading cached data...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        meta = np.load(meta_cache, allow_pickle=True).item()
    else:
        print("Processing data from scratch...")

        # Load metadata-defined splits
        train_path = os.path.join(METADATA_DIR, "train.csv")
        val_path = os.path.join(METADATA_DIR, "val.csv")
        test_path = os.path.join(METADATA_DIR, "test.csv")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        # Mark splits for later separation
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        # Combine for transductive operations
        full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

        # Feature Engineering: f_27
        # 1. Unique character count
        full_df["f_27_unique"] = full_df["f_27"].apply(lambda x: len(set(x)))

        # 2. String decomposition (10 chars)
        # Assuming f_27 is always length 10 based on dataset analysis
        for i in range(10):
            full_df[f"f_27_{i}"] = full_df["f_27"].str[i]

        # Define Column Groups
        # Categorical: f_29, f_30, and f_27 parts
        cat_cols = [f"f_27_{i}" for i in range(10)] + ["f_29", "f_30"]

        # Continuous: All f_ columns excluding cat_cols and f_27 (raw string)
        # Also exclude id, target, source_path, split
        exclude_cols = ["id", "target", "source_path", "split", "f_27"] + cat_cols
        cont_cols = [c for c in full_df.columns if c not in exclude_cols]

        # Transductive Label Encoding
        vocab_sizes = {}
        encoder = OrdinalEncoder(dtype=np.int64)
        full_df[cat_cols] = encoder.fit_transform(full_df[cat_cols])

        for col in cat_cols:
            vocab_sizes[col] = int(full_df[col].max() + 1)

        # Scaling
        # Fit only on training set to prevent data leakage
        scaler = StandardScaler()
        train_mask = full_df["split"] == "train"
        scaler.fit(full_df.loc[train_mask, cont_cols])
        full_df[cont_cols] = scaler.transform(full_df[cont_cols])

        # Convert to float32 for continuous features to save memory/compute
        full_df[cont_cols] = full_df[cont_cols].astype(np.float32)

        # Split back into respective dataframes
        train_df = (
            full_df[full_df["split"] == "train"]
            .drop(columns=["split", "f_27", "source_path"])
            .reset_index(drop=True)
        )
        val_df = (
            full_df[full_df["split"] == "val"]
            .drop(columns=["split", "f_27", "source_path"])
            .reset_index(drop=True)
        )
        test_df = (
            full_df[full_df["split"] == "test"]
            .drop(columns=["split", "f_27", "source_path", "target"])
            .reset_index(drop=True)
        )

        # Metadata for model initialization
        meta = {
            "cat_cols": cat_cols,
            "cont_cols": cont_cols,
            "vocab_sizes": vocab_sizes,
        }

        # Save to cache
        train_df.to_parquet(train_cache)
        val_df.to_parquet(val_cache)
        test_df.to_parquet(test_cache)
        np.save(meta_cache, meta)

    # Handle debugging (subsampling)
    if debug_rows is not None:
        print(f"Debugging mode: subsampling to {debug_rows} rows.")
        train_df = train_df.iloc[:debug_rows]
        val_df = val_df.iloc[:debug_rows]
        test_df = test_df.iloc[:debug_rows]

    # Create Datasets
    cat_cols = meta["cat_cols"]
    cont_cols = meta["cont_cols"]

    train_dataset = ManufacturingDataset(train_df, cat_cols, cont_cols)
    val_dataset = ManufacturingDataset(val_df, cat_cols, cont_cols)
    test_dataset = ManufacturingDataset(test_df, cat_cols, cont_cols, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, meta
