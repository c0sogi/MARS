import pandas as pd
import numpy as np
import os
import random


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    # Load transactions with optimized types
    transactions = pd.read_csv(
        os.path.join(INPUT_DIR, "transactions_train.csv"),
        dtype={"article_id": "int32", "price": "float32", "sales_channel_id": "int8"},
    )

    # Load sample submission to define the test set users
    sample_submission = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

    print(
        f"Loaded {len(transactions)} transactions and {len(sample_submission)} test customers."
    )

    # --- 1. Generate Image Paths ---
    print("Generating image paths...")
    # Get unique articles to process path generation efficiently
    unique_articles = transactions["article_id"].unique()
    article_df = pd.DataFrame({"article_id": unique_articles})

    def get_rel_path(article_id):
        # Format: 0108775015 (10 digits) -> folder 010 -> file 0108775015.jpg
        s = str(article_id).zfill(10)
        folder = s[:3]
        return f"images/{folder}/{s}.jpg"

    article_df["image_path"] = article_df["article_id"].apply(get_rel_path)

    # Merge image paths back to transactions
    transactions = transactions.merge(article_df, on="article_id", how="left")

    # --- 2. Group Sampling Split ---
    print("Performing Group Sampling split (80/20 by customer_id)...")
    unique_customers = transactions["customer_id"].unique()
    n_customers = len(unique_customers)

    # Shuffle customers
    rng = np.random.RandomState(RANDOM_STATE)
    shuffled_customers = rng.permutation(unique_customers)

    # Determine split index
    split_idx = int(n_customers * 0.8)
    train_users_set = set(shuffled_customers[:split_idx])
    val_users_set = set(shuffled_customers[split_idx:])

    # Split transactions
    # Using boolean indexing with isin is efficient
    is_train = transactions["customer_id"].isin(train_users_set)
    train_df = transactions[is_train]
    val_df = transactions[~is_train]

    print(f"Train transactions: {len(train_df)}")
    print(f"Val transactions: {len(val_df)}")

    # --- 3. Save Metadata ---
    print("Saving metadata files...")
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    # Test metadata is just the list of customers we need to predict for
    sample_submission[["customer_id"]].to_csv(test_path, index=False)

    # --- 4. Validation & Checks ---
    print("\nPerforming validation checks...")

    # Reload data to verify disk persistence
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # A. Summary Statistics
    print("\n--- Summary Statistics ---")
    datasets = {"Train": df_train, "Validation": df_val, "Test": df_test}
    for name, df in datasets.items():
        print(f"Dataset: {name}")
        print(f"  Shape: {df.shape}")
        if "customer_id" in df.columns:
            print(f"  Unique Customers: {df['customer_id'].nunique()}")
        if "article_id" in df.columns:
            print(f"  Unique Articles: {df['article_id'].nunique()}")
            print(
                f"  Class Distribution (Top 3): \n{df['article_id'].value_counts().head(3)}"
            )
        print("-" * 30)

    # B. File Path Verification
    print("\n--- Verifying File Paths ---")

    def verify_paths(df, dataset_name):
        if "image_path" not in df.columns:
            return

        # Sample 1000 paths
        sample_size = min(1000, len(df))
        sample_paths = (
            df["image_path"].dropna().sample(n=sample_size, random_state=RANDOM_STATE)
        )

        missing_count = 0
        missing_examples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(
            f"[{dataset_name}] Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_count > 0:
            print(f"[{dataset_name}] Example missing paths: {missing_examples}")

        if missing_ratio > 0.5:
            raise RuntimeError(
                f"Missing file ratio {missing_ratio} exceeds 0.5 for {dataset_name} dataset."
            )

    verify_paths(df_train, "Train")
    verify_paths(df_val, "Validation")

    # C. Split Requirements Verification
    print("\n--- Verifying Split Requirements ---")
    train_users = set(df_train["customer_id"].unique())
    val_users = set(df_val["customer_id"].unique())

    # Check 1: No overlap (Group Sampling)
    intersection = train_users.intersection(val_users)
    print(f"User overlap count: {len(intersection)}")
    if len(intersection) > 0:
        raise AssertionError(
            f"Group split failed: {len(intersection)} users found in both train and validation sets."
        )

    # Check 2: Ratio
    total_split_users = len(train_users) + len(val_users)
    actual_train_ratio = len(train_users) / total_split_users
    print(f"Actual Train Ratio: {actual_train_ratio:.4f}")

    # Tolerance of 1%
    if not (0.79 <= actual_train_ratio <= 0.81):
        raise AssertionError(
            f"Split ratio {actual_train_ratio:.4f} is not approximately 0.80"
        )

    print("\nAll checks passed successfully. Metadata generation complete.")


if __name__ == "__main__":
    main()
