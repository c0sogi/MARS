import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config, set_seed, ManufacturingDataset


def preprocess_data(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Loads, preprocesses, and caches data.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader, vocab_sizes, num_cont_features)
    """
    set_seed()

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_processed.parquet")
    meta_cache = os.path.join(Config.WORKING_DIR, "metadata.npy")

    # Define feature groups explicitly to ensure consistent ordering across runs
    # Categorical: f_27 characters (0-9), f_29, f_30
    cat_features_base = [f"f_27_{i}" for i in range(10)] + ["f_29", "f_30"]

    # Continuous: f_00 to f_28 (excluding f_27), plus unique_char_count
    cont_features_base = [f"f_{i:02d}" for i in range(29) if i != 27] + [
        "unique_char_count"
    ]

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(meta_cache)
    ):
        print("Loading cached data...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache)
        metadata = np.load(meta_cache, allow_pickle=True).item()
        vocab_sizes = metadata["vocab_sizes"]

        # Identify columns based on the STRICT ORDER defined by cat_features_base
        # Do NOT rely on df.columns iteration which may vary based on concat order
        cat_cols = [f"cat_{c}" for c in cat_features_base]
        cont_cols = [f"cont_{c}" for c in cont_features_base]

        # Create Datasets
        train_ds = ManufacturingDataset(
            df_train[cat_cols].values,
            df_train[cont_cols].values,
            df_train["target"].values,
        )
        val_ds = ManufacturingDataset(
            df_val[cat_cols].values, df_val[cont_cols].values, df_val["target"].values
        )
        test_ds = ManufacturingDataset(
            df_test[cat_cols].values, df_test[cont_cols].values, None
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader, vocab_sizes, len(cont_cols)

    # 2. Process from Scratch
    print("Processing data from scratch...")

    # Load raw metadata CSVs
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Mark splits for later separation
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    # Concatenate for transductive processing
    full_df = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)

    # --- Feature Engineering ---

    # 1. f_27 decomposition: Split string into characters
    # We expect f_27 to be a string of length 10
    chars = full_df["f_27"].apply(lambda x: list(str(x)))
    char_df = pd.DataFrame(chars.tolist(), columns=[f"f_27_{i}" for i in range(10)])

    # 2. Unique character count
    full_df["unique_char_count"] = full_df["f_27"].apply(lambda x: len(set(str(x))))

    # Add char columns to full_df
    full_df = pd.concat([full_df, char_df], axis=1)

    # Use the base lists defined at the top
    cat_features = cat_features_base
    cont_features = cont_features_base

    # --- Encoding Categoricals (Transductive) ---
    print("Encoding categorical features...")
    ord_enc = OrdinalEncoder(
        dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
    )
    full_df[cat_features] = ord_enc.fit_transform(full_df[cat_features].astype(str))

    # Calculate vocab sizes (max index + 1)
    vocab_sizes = full_df[cat_features].max().astype(int).values + 1

    # --- Splitting Back ---
    train_proc = full_df[full_df["split"] == "train"].copy()
    val_proc = full_df[full_df["split"] == "val"].copy()
    test_proc = full_df[full_df["split"] == "test"].copy()

    # --- Scaling Continuous (Fit on Train Only) ---
    print("Scaling continuous features...")
    scaler = StandardScaler()

    # Fit on Train
    scaler.fit(train_proc[cont_features])

    # Transform all
    train_proc[cont_features] = scaler.transform(train_proc[cont_features])
    val_proc[cont_features] = scaler.transform(val_proc[cont_features])
    test_proc[cont_features] = scaler.transform(test_proc[cont_features])

    # --- Final Formatting ---
    # Rename columns for clarity in cache and easy identification
    rename_map = {c: f"cat_{c}" for c in cat_features}
    rename_map.update({c: f"cont_{c}" for c in cont_features})

    train_proc = train_proc.rename(columns=rename_map)
    val_proc = val_proc.rename(columns=rename_map)
    test_proc = test_proc.rename(columns=rename_map)

    final_cat_cols = [f"cat_{c}" for c in cat_features]
    final_cont_cols = [f"cont_{c}" for c in cont_features]

    # Save to cache
    print("Saving to cache...")
    train_proc.to_parquet(train_cache)
    val_proc.to_parquet(val_cache)
    test_proc.to_parquet(test_cache)

    metadata = {"vocab_sizes": vocab_sizes}
    np.save(meta_cache, metadata)

    # Create Datasets & Loaders
    train_ds = ManufacturingDataset(
        train_proc[final_cat_cols].values,
        train_proc[final_cont_cols].values,
        train_proc["target"].values,
    )
    val_ds = ManufacturingDataset(
        val_proc[final_cat_cols].values,
        val_proc[final_cont_cols].values,
        val_proc["target"].values,
    )
    test_ds = ManufacturingDataset(
        test_proc[final_cat_cols].values, test_proc[final_cont_cols].values, None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, vocab_sizes, len(final_cont_cols)
