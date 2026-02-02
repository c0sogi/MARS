import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Constants
TRAIN_PATH = "./metadata/train.csv"
VAL_PATH = "./metadata/val.csv"
TEST_PATH = "./metadata/test.csv"
CACHE_DIR = "./working/idea_18/"


def feature_engineering(df):
    """
    Decomposes f_27 into character columns and computes unique character count.
    """
    # Avoid modifying original dataframe
    df = df.copy()

    # Split f_27 into 10 separate character columns
    # We assume f_27 exists and is a string of length 10
    if "f_27" in df.columns:
        for i in range(10):
            df[f"ch_{i}"] = df["f_27"].str[i]

        # Compute unique character count
        df["unique_char_count"] = df["f_27"].apply(lambda x: len(set(x)))

        # Drop the original f_27 column
        df = df.drop(columns=["f_27"])

    return df


def prepare_processors(train_df, val_df, test_df):
    """
    Fits encoders and scalers.
    - Categorical: Transductive fit (Train + Val + Test)
    - Continuous: Fit on Train only
    """
    # Identify columns
    # Based on strategy: f_29, f_30 and ch_0...ch_9 are categorical
    cat_cols = ["f_29", "f_30"] + [f"ch_{i}" for i in range(10)]

    # Continuous are everything else except metadata
    exclude_cols = ["id", "target", "source_path", "split"] + cat_cols
    cont_cols = [c for c in train_df.columns if c not in exclude_cols]

    # --- Categorical Processing (Transductive) ---
    # Concatenate all for vocabulary alignment
    combined_cat = pd.concat(
        [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
    )

    enc = OrdinalEncoder(
        dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
    )
    enc.fit(combined_cat)

    # Transform
    train_cat = enc.transform(train_df[cat_cols])
    val_cat = enc.transform(val_df[cat_cols])
    test_cat = enc.transform(test_df[cat_cols])

    # Calculate vocab sizes (max index + 1)
    # We use the combined data to ensure we capture the max index across all sets
    all_cat_encoded = enc.transform(combined_cat)
    vocab_sizes = all_cat_encoded.max(axis=0) + 1
    vocab_sizes = vocab_sizes.astype(int).tolist()

    # --- Continuous Processing (Fit on Train) ---
    scaler = StandardScaler()
    scaler.fit(train_df[cont_cols])

    train_cont = scaler.transform(train_df[cont_cols])
    val_cont = scaler.transform(val_df[cont_cols])
    test_cont = scaler.transform(test_df[cont_cols])

    return (
        (train_cat, train_cont),
        (val_cat, val_cont),
        (test_cat, test_cont),
        vocab_sizes,
    )


class ManufacturingDataset(Dataset):
    def __init__(self, x_cat, x_cont, y=None):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self):
        return len(self.x_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_cat[idx], self.x_cont[idx], self.y[idx]
        return self.x_cat[idx], self.x_cont[idx]


def process_data(load_cached_data=True):
    """
    Orchestrates data loading, feature engineering, preprocessing, and caching.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_train_cat": os.path.join(CACHE_DIR, "X_train_cat.npy"),
        "X_train_cont": os.path.join(CACHE_DIR, "X_train_cont.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_val_cat": os.path.join(CACHE_DIR, "X_val_cat.npy"),
        "X_val_cont": os.path.join(CACHE_DIR, "X_val_cont.npy"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "X_test_cat": os.path.join(CACHE_DIR, "X_test_cat.npy"),
        "X_test_cont": os.path.join(CACHE_DIR, "X_test_cont.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        "vocab_sizes": os.path.join(CACHE_DIR, "vocab_sizes.npy"),
    }

    # 1. Check Cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            data = {k: np.load(v, allow_pickle=True) for k, v in cache_files.items()}
            # Convert vocab_sizes back to list
            data["vocab_sizes"] = data["vocab_sizes"].tolist()
            return data

    print("Processing data from scratch...")

    # 2. Load Raw Data
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Extract Targets
    y_train = df_train["target"].values.astype(np.float32)
    y_val = df_val["target"].values.astype(np.float32)
    test_ids = df_test["id"].values

    # 3. Feature Engineering
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    # 4. Preprocessing (Encoding & Scaling)
    (train_cat, train_cont), (val_cat, val_cont), (test_cat, test_cont), vocab_sizes = (
        prepare_processors(df_train, df_val, df_test)
    )

    # 5. Save to Cache
    np.save(cache_files["X_train_cat"], train_cat.astype(np.int64))
    np.save(cache_files["X_train_cont"], train_cont.astype(np.float32))
    np.save(cache_files["y_train"], y_train)

    np.save(cache_files["X_val_cat"], val_cat.astype(np.int64))
    np.save(cache_files["X_val_cont"], val_cont.astype(np.float32))
    np.save(cache_files["y_val"], y_val)

    np.save(cache_files["X_test_cat"], test_cat.astype(np.int64))
    np.save(cache_files["X_test_cont"], test_cont.astype(np.float32))

    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["vocab_sizes"], np.array(vocab_sizes))

    return {
        "X_train_cat": train_cat.astype(np.int64),
        "X_train_cont": train_cont.astype(np.float32),
        "y_train": y_train,
        "X_val_cat": val_cat.astype(np.int64),
        "X_val_cont": val_cont.astype(np.float32),
        "y_val": y_val,
        "X_test_cat": test_cat.astype(np.int64),
        "X_test_cont": test_cont.astype(np.float32),
        "test_ids": test_ids,
        "vocab_sizes": vocab_sizes,
    }


def get_dataloaders(data, batch_size=1024, num_workers=4):
    """
    Creates PyTorch DataLoaders for train, val, and test sets.
    """
    train_dataset = ManufacturingDataset(
        data["X_train_cat"], data["X_train_cont"], data["y_train"]
    )

    val_dataset = ManufacturingDataset(
        data["X_val_cat"], data["X_val_cont"], data["y_val"]
    )

    test_dataset = ManufacturingDataset(
        data["X_test_cat"], data["X_test_cont"], None  # No target for test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
