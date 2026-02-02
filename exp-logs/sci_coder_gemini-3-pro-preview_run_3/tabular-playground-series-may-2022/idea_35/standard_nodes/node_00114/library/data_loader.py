import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import METADATA_DIR, WORKING_DIR, SEED

# Set global seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing control data.
    Yields (categorical_inputs, continuous_inputs, target) or (categorical_inputs, continuous_inputs).
    """

    def __init__(self, cat_features, cont_features, targets=None):
        self.cat_features = torch.tensor(cat_features, dtype=torch.long)
        self.cont_features = torch.tensor(cont_features, dtype=torch.float32)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.cat_features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.cat_features[idx], self.cont_features[idx], self.targets[idx]
        else:
            return self.cat_features[idx], self.cont_features[idx]


def engineer_features(df):
    """
    Performs feature engineering:
    1. Computes 'unique_character_count' from 'f_27'.
    2. Decomposes 'f_27' into 10 separate character columns.
    3. Drops the original 'f_27'.
    """
    if "f_27" in df.columns:
        # Vectorized processing for speed
        s_values = df["f_27"].astype(str).values

        # 1. Set Cardinality: unique_character_count
        unique_counts = [len(set(s)) for s in s_values]
        df["unique_character_count"] = unique_counts

        # 2. String Decomposition into 10 fixed columns
        # f_27 is known to be length 10
        chars = [[c for c in s] for s in s_values]
        char_cols = [f"f_27_{i}" for i in range(10)]
        char_df = pd.DataFrame(chars, columns=char_cols, index=df.index)

        # Concatenate and drop original
        df = pd.concat([df, char_df], axis=1)
        df = df.drop(columns=["f_27"])

    return df


def prepare_data(load_cached_data=True):
    """
    Orchestrates data loading, feature engineering, and transductive preprocessing.
    Implements caching to ./working/idea_35/ using .npz files.

    Returns:
        train_dataset (ManufacturingDataset)
        val_dataset (ManufacturingDataset)
        test_dataset (ManufacturingDataset)
        vocab_sizes (np.array): Array of vocabulary sizes for categorical features.
    """

    # Define cache paths
    cache_dir = WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.npz")
    val_cache = os.path.join(cache_dir, "val_processed.npz")
    test_cache = os.path.join(cache_dir, "test_processed.npz")
    vocab_cache = os.path.join(cache_dir, "vocab_sizes.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(vocab_cache)
        ):
            print("Loading cached processed data...")
            train_data = np.load(train_cache)
            val_data = np.load(val_cache)
            test_data = np.load(test_cache)
            vocab_sizes = np.load(vocab_cache)

            train_ds = ManufacturingDataset(
                train_data["cat"], train_data["cont"], train_data["target"]
            )
            val_ds = ManufacturingDataset(
                val_data["cat"], val_data["cont"], val_data["target"]
            )
            test_ds = ManufacturingDataset(test_data["cat"], test_data["cont"])

            return train_ds, val_ds, test_ds, vocab_sizes

    print("Processing data from scratch...")

    # 2. Load Raw Data using Metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Extract targets and drop non-feature columns
    y_train = train_df["target"].values
    y_val = val_df["target"].values

    drop_cols = ["id", "source_path", "target"]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    val_df = val_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=["id", "source_path"], errors="ignore")

    # 3. Feature Engineering
    print("Engineering features...")
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    # 4. Identify Columns
    # Categorical: f_29, f_30, and decomposed f_27 columns
    # Continuous: All others

    # Find decomposed columns
    f27_cols = [c for c in train_df.columns if c.startswith("f_27_")]
    f27_cols.sort()  # Ensure deterministic order

    cat_cols = ["f_29", "f_30"] + f27_cols
    # Filter to ensure they exist
    cat_cols = [c for c in cat_cols if c in train_df.columns]

    cont_cols = [c for c in train_df.columns if c not in cat_cols]

    # 5. Transductive Preprocessing

    # Categorical: Fit OrdinalEncoder on ALL data (Train + Val + Test)
    print("Encoding categorical features (Transductive)...")
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )

    all_cat = pd.concat(
        [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
    )
    encoder.fit(all_cat)

    X_cat_train = encoder.transform(train_df[cat_cols])
    X_cat_val = encoder.transform(val_df[cat_cols])
    X_cat_test = encoder.transform(test_df[cat_cols])

    # Calculate vocab sizes for embeddings
    vocab_sizes = np.array([len(cats) for cats in encoder.categories_])

    # Continuous: Fit StandardScaler on TRAIN only
    print("Scaling continuous features...")
    scaler = StandardScaler()
    scaler.fit(train_df[cont_cols])

    X_cont_train = scaler.transform(train_df[cont_cols]).astype(np.float32)
    X_cont_val = scaler.transform(val_df[cont_cols]).astype(np.float32)
    X_cont_test = scaler.transform(test_df[cont_cols]).astype(np.float32)

    # 6. Save to Cache
    print("Saving processed data to cache...")
    np.savez(train_cache, cat=X_cat_train, cont=X_cont_train, target=y_train)
    np.savez(val_cache, cat=X_cat_val, cont=X_cont_val, target=y_val)
    np.savez(test_cache, cat=X_cat_test, cont=X_cont_test)
    np.save(vocab_cache, vocab_sizes)

    # 7. Create Datasets
    train_ds = ManufacturingDataset(X_cat_train, X_cont_train, y_train)
    val_ds = ManufacturingDataset(X_cat_val, X_cont_val, y_val)
    test_ds = ManufacturingDataset(X_cat_test, X_cont_test)

    return train_ds, val_ds, test_ds, vocab_sizes
