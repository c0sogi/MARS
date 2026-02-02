import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path

# Configuration
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
RANDOM_STATE = 42


def generate_image_path(article_id):
    """
    Converts article_id (int or str) to relative image path.
    Format: images/xxx/0xxxxxxxx.jpg
    """
    s = str(article_id).zfill(10)
    folder = s[:3]
    return f"images/{folder}/{s}.jpg"


def run():
    print("Starting metadata generation...")

    # 1. Setup Directories
    if METADATA_DIR.exists():
        shutil.rmtree(METADATA_DIR)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load Raw Data
    print("Loading raw datasets...")
    # Read articles to get image mapping
    articles_df = pd.read_csv(INPUT_DIR / "articles.csv")

    # Read transactions
    transactions_df = pd.read_csv(INPUT_DIR / "transactions_train.csv")

    # Read sample submission for test customers
    sample_sub_df = pd.read_csv(INPUT_DIR / "sample_submission.csv")

    # 3. Generate Image Paths and Merge
    print("Generating image paths...")
    # Vectorized string operations for performance
    # Ensure article_id is treated as string with leading zeros
    aid_str = articles_df["article_id"].astype(str).str.zfill(10)
    subfolders = aid_str.str[:3]
    filenames = aid_str + ".jpg"

    # Create path column relative to ./input
    articles_df["image_path"] = "images/" + subfolders + "/" + filenames

    # Create a mapping dictionary or smaller df to merge
    article_map = articles_df[["article_id", "image_path"]]

    print("Merging image paths into transactions...")
    # Merge image paths into transactions
    # transactions_df has 'article_id' as int64 usually, articles_df also int64.
    # The merge key types must match.
    transactions_df = transactions_df.merge(article_map, on="article_id", how="left")

    # 4. Group Split (80/20 by Customer)
    print("Performing Group Shuffle Split...")
    unique_customers = transactions_df["customer_id"].unique()

    # Random Shuffle
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_customers)

    # Split indices
    n_customers = len(unique_customers)
    n_train = int(n_customers * 0.8)

    train_cust_ids = set(unique_customers[:n_train])
    val_cust_ids = set(unique_customers[n_train:])

    # Create a mapping for fast filtering
    # We create a dataframe of customers with their assigned split
    print("Assigning splits...")
    cust_split_df = pd.DataFrame(
        {
            "customer_id": unique_customers,
            "split_group": ["train"] * n_train + ["val"] * (n_customers - n_train),
        }
    )

    # Merge split info back to transactions
    # This is more memory efficient than .isin() on large arrays
    transactions_df = transactions_df.merge(
        cust_split_df, on="customer_id", how="inner"
    )

    # Partition Data
    train_df = transactions_df[transactions_df["split_group"] == "train"].drop(
        columns=["split_group"]
    )
    val_df = transactions_df[transactions_df["split_group"] == "val"].drop(
        columns=["split_group"]
    )

    # 5. Save Metadata
    print("Saving metadata files...")
    train_path = METADATA_DIR / "train_metadata.parquet"
    val_path = METADATA_DIR / "val_metadata.parquet"
    test_path = METADATA_DIR / "test_metadata.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    sample_sub_df.to_parquet(test_path, index=False)

    print("Metadata generation complete.")

    # 6. Verification and Checks
    print("Running verification checks...")

    # Load back data to verify
    train_check = pd.read_parquet(train_path)
    val_check = pd.read_parquet(val_path)
    test_check = pd.read_parquet(test_path)

    # Summary Statistics
    print("=" * 30)
    print("DATASET SUMMARY")
    print("=" * 30)
    print(f"Train Samples: {len(train_check)}")
    print(f"Val Samples:   {len(val_check)}")
    print(f"Test Samples:  {len(test_check)}")
    print(f"Train Unique Users: {train_check['customer_id'].nunique()}")
    print(f"Val Unique Users:   {val_check['customer_id'].nunique()}")

    # Check Split Ratio (User level)
    n_train_users = train_check["customer_id"].nunique()
    n_val_users = val_check["customer_id"].nunique()
    total_users = n_train_users + n_val_users
    print(
        f"Split Ratio (Users): Train={n_train_users/total_users:.4f}, Val={n_val_users/total_users:.4f}"
    )

    # Verify Group Split (No leakage)
    train_users_set = set(train_check["customer_id"].unique())
    val_users_set = set(val_check["customer_id"].unique())
    intersection = train_users_set.intersection(val_users_set)

    if len(intersection) > 0:
        raise AssertionError(
            f"Data Leakage detected! {len(intersection)} users found in both Train and Val sets."
        )
    print("Split Verification: PASSED (No user overlap)")

    # Check File Paths
    print("Checking file paths...")
    # Combine train and val to sample paths
    all_paths = pd.concat([train_check["image_path"], val_check["image_path"]]).dropna()

    if len(all_paths) > 0:
        sample_paths = all_paths.sample(
            n=min(1000, len(all_paths)), random_state=RANDOM_STATE
        )

        missing_count = 0
        missing_examples = []

        for p in sample_paths:
            full_path = INPUT_DIR / p
            if not full_path.exists():
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(str(p))

        missing_ratio = missing_count / len(sample_paths)
        print(
            f"Missing File Ratio: {missing_ratio:.4f} ({missing_count}/{len(sample_paths)})"
        )

        if len(missing_examples) > 0:
            print("Example missing paths:")
            for mp in missing_examples:
                print(f" - {mp}")

        if missing_ratio > 0.5:
            raise FileNotFoundError(
                f"Error: {missing_ratio:.2%} of sampled image paths do not exist (Threshold: 50%)."
            )

        print("File Path Verification: PASSED")
    else:
        print("No image paths found to verify.")


if __name__ == "__main__":
    run()
