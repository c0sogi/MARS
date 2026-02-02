import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Constants
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_image_path(article_id_series):
    """
    Vectorized generation of image paths from article_ids.
    article_id is int64, needs to be zero-padded to 10 chars.
    Format: images/xxx/xxxyyyyyyy.jpg
    """
    # Convert to string and pad with zeros to length 10
    ids = article_id_series.astype(str).str.zfill(10)
    # Construct path: images/ + first 3 chars + / + full id + .jpg
    return "images/" + ids.str.slice(0, 3) + "/" + ids + ".jpg"


def check_paths(df, dataset_name, base_dir):
    """
    Checks if a random sample of file paths in the dataframe exist.
    """
    if "image_path" not in df.columns:
        print(
            f"[{dataset_name}] No 'image_path' column found. Skipping path verification."
        )
        return

    sample_size = 1000
    if len(df) < sample_size:
        sample = df
    else:
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        rel_path = row["image_path"]
        full_path = base_dir / rel_path
        if not full_path.exists():
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    ratio = missing_count / len(sample)
    print(
        f"[{dataset_name}] Missing file ratio: {ratio:.4f} ({missing_count}/{len(sample)})"
    )

    if len(missing_samples) > 0:
        print(f"[{dataset_name}] Example missing paths: {missing_samples}")

    if ratio > 0.5:
        raise FileNotFoundError(
            f"[{dataset_name}] More than 50% of image paths are missing!"
        )


def main():
    # 1. Setup
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    # Load transactions (Training Data)
    # Using pyarrow engine for speed if available, otherwise default
    try:
        transactions = pd.read_csv(
            INPUT_DIR / "transactions_train.csv", dtype={"article_id": str}
        )
    except Exception:
        # Fallback if dtype str fails or other issue, though article_id should be read as object or int
        transactions = pd.read_csv(INPUT_DIR / "transactions_train.csv")

    # Load sample submission (Test Data definition)
    sample_submission = pd.read_csv(INPUT_DIR / "sample_submission.csv")

    print(f"Total transactions: {len(transactions)}")
    print(f"Total submission customers: {len(sample_submission)}")

    # 2. Split Data (Group Sampling by customer_id)
    print("Performing Group Split (80/20) by customer_id...")
    unique_customers = transactions["customer_id"].unique()

    # Shuffle customers
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_customers)

    # Split
    n_train = int(len(unique_customers) * (1 - VAL_SIZE))
    train_customers = set(unique_customers[:n_train])
    val_customers = set(unique_customers[n_train:])

    # Filter transactions
    # We use boolean indexing. To speed up, we can map customer_id to a boolean
    print("Filtering transactions for split...")
    # Create a mask
    train_mask = transactions["customer_id"].isin(train_customers)
    train_df = transactions[train_mask].copy()
    val_df = transactions[~train_mask].copy()

    # 3. Generate Metadata (Add image paths)
    print("Generating image paths...")
    # Ensure article_id is treated correctly for path generation
    # If read as int, convert. If read as str, ensure padding.

    # Helper to process df
    def process_df(df):
        # Ensure article_id is string
        if not pd.api.types.is_string_dtype(df["article_id"]):
            df["article_id"] = df["article_id"].astype(str)

        # Remove .0 if it appeared from float conversion
        df["article_id"] = df["article_id"].apply(
            lambda x: x.replace(".0", "") if x.endswith(".0") else x
        )

        df["image_path"] = generate_image_path(df["article_id"])
        return df

    train_df = process_df(train_df)
    val_df = process_df(val_df)

    # Test metadata is just the customers we need to predict for
    test_df = sample_submission[["customer_id"]].copy()
    # Test data doesn't have 'article_id' ground truth for the prediction period, so no image paths generated here.

    # 4. Save Metadata
    print("Saving metadata to Parquet...")
    train_path = METADATA_DIR / "train.parquet"
    val_path = METADATA_DIR / "val.parquet"
    test_path = METADATA_DIR / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # Free memory
    del transactions, train_df, val_df, test_df
    import gc

    gc.collect()

    # 5. Verification
    print("\nVerifying generated metadata...")

    # Load back
    train_meta = pd.read_parquet(train_path)
    val_meta = pd.read_parquet(val_path)
    test_meta = pd.read_parquet(test_path)

    # Stats
    print("=== Summary Statistics ===")
    print(
        f"Train Set: {len(train_meta)} rows, {train_meta['customer_id'].nunique()} unique customers"
    )
    print(
        f"Val Set:   {len(val_meta)} rows, {val_meta['customer_id'].nunique()} unique customers"
    )
    print(
        f"Test Set:  {len(test_meta)} rows, {test_meta['customer_id'].nunique()} unique customers"
    )

    # Validate Split
    print("\nVerifying Group Split...")
    train_cust_set = set(train_meta["customer_id"].unique())
    val_cust_set = set(val_meta["customer_id"].unique())
    intersection = train_cust_set.intersection(val_cust_set)

    if len(intersection) > 0:
        raise AssertionError(
            f"Split failed! Found {len(intersection)} customers in both Train and Validation sets."
        )
    print("Split verification passed: No customer overlap.")

    # Validate Ratios
    total_cust = len(train_cust_set) + len(val_cust_set)
    train_ratio = len(train_cust_set) / total_cust
    print(f"Train Split Ratio (by customer): {train_ratio:.4f} (Target: ~0.8)")

    # Validate File Paths
    print("\nVerifying File Paths...")
    check_paths(train_meta, "Train", INPUT_DIR)
    check_paths(val_meta, "Validation", INPUT_DIR)
    check_paths(test_meta, "Test", INPUT_DIR)  # Likely skips

    print("\nMetadata generation and verification complete.")


if __name__ == "__main__":
    main()
