import pandas as pd
import numpy as np
import os


def generate_metadata():
    # Constants
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_FILE = os.path.join(INPUT_DIR, "en_train.csv")
    TEST_FILE = os.path.join(INPUT_DIR, "en_test_2.csv")
    RANDOM_STATE = 42

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw training data...")
    # Read csv with na_filter=False to prevent 'nan', 'null' etc from becoming NaN
    df_train_raw = pd.read_csv(
        TRAIN_FILE, dtype={"sentence_id": int, "token_id": int}, keep_default_na=False
    )

    # Create 'id' column
    df_train_raw["id"] = (
        df_train_raw["sentence_id"].astype(str)
        + "_"
        + df_train_raw["token_id"].astype(str)
    )

    # Group Sampling Split
    print("Performing group split on sentence_id...")
    unique_sentence_ids = df_train_raw["sentence_id"].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(unique_sentence_ids)

    n_samples = len(unique_sentence_ids)
    n_train = int(n_samples * 0.8)

    train_sentence_ids = set(unique_sentence_ids[:n_train])
    val_sentence_ids = set(unique_sentence_ids[n_train:])

    # Create masks
    # Using map is faster than isin for large datasets if we map to a boolean
    # But given 9M rows, isin with a set is reasonably optimized in pandas.
    # Let's use a merge/join approach or map for efficiency if needed, but isin(set) is usually fine.
    # To be extremely efficient with 9M rows:
    # Create a dataframe for split mapping
    split_map = pd.DataFrame(
        {
            "sentence_id": np.concatenate(
                [list(train_sentence_ids), list(val_sentence_ids)]
            ),
            "split_group": ["train"] * len(train_sentence_ids)
            + ["val"] * len(val_sentence_ids),
        }
    )

    # Merge to assign split
    df_train_raw = df_train_raw.merge(split_map, on="sentence_id", how="left")

    train_df = df_train_raw[df_train_raw["split_group"] == "train"].drop(
        columns=["split_group"]
    )
    val_df = df_train_raw[df_train_raw["split_group"] == "val"].drop(
        columns=["split_group"]
    )

    print(f"Saving training metadata ({len(train_df)} rows)...")
    train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)

    print(f"Saving validation metadata ({len(val_df)} rows)...")
    val_df.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)

    # Cleanup memory
    del df_train_raw, train_df, val_df, split_map, unique_sentence_ids

    print("Loading and processing test data...")
    df_test_raw = pd.read_csv(
        TEST_FILE, dtype={"sentence_id": int, "token_id": int}, keep_default_na=False
    )
    df_test_raw["id"] = (
        df_test_raw["sentence_id"].astype(str)
        + "_"
        + df_test_raw["token_id"].astype(str)
    )

    print(f"Saving test metadata ({len(df_test_raw)} rows)...")
    df_test_raw.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)
    del df_test_raw

    # ==========================================
    # Verification Step
    # ==========================================
    print("\nVerifying generated metadata...")

    # Load datasets
    train_meta = pd.read_csv(
        os.path.join(METADATA_DIR, "train.csv"), keep_default_na=False
    )
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"), keep_default_na=False)
    test_meta = pd.read_csv(
        os.path.join(METADATA_DIR, "test.csv"), keep_default_na=False
    )

    # 1. Summary Statistics
    print("-" * 30)
    print("Summary Statistics")
    print("-" * 30)

    datasets = {"Train": train_meta, "Validation": val_meta, "Test": test_meta}

    for name, df in datasets.items():
        print(f"\nDataset: {name}")
        print(f"Shape: {df.shape}")
        print(f"Unique Sentences: {df['sentence_id'].nunique()}")
        if "class" in df.columns:
            print(
                f"Class Distribution:\n{df['class'].value_counts(normalize=True).head()}"
            )
        else:
            print("Class Distribution: N/A (Test Set)")

    # 2. File Path Check
    # The dataset contains text data directly, not file paths.
    # We skip the "missing file ratio" check or treat it as 0 missing.
    # However, to strictly follow the prompt structure, if there were file paths, we would check them.
    # Here we explicitly state no external files are referenced.
    print(
        "\nChecking file paths: No external file paths in metadata (text contained in CSV)."
    )

    # 3. Validation Split Verification
    print("\nVerifying Validation Split Requirements...")

    train_sents = set(train_meta["sentence_id"].unique())
    val_sents = set(val_meta["sentence_id"].unique())

    # Assert stratification/group split success (No overlap)
    intersection = train_sents.intersection(val_sents)
    if len(intersection) > 0:
        raise AssertionError(
            f"Data Leakage detected! {len(intersection)} sentences found in both Train and Validation sets."
        )

    # Check ratio roughly 80:20
    total_sents = len(train_sents) + len(val_sents)
    train_ratio = len(train_sents) / total_sents
    print(f"Split Ratio (Sentences): Train={train_ratio:.4f}, Val={1-train_ratio:.4f}")

    if not (0.79 < train_ratio < 0.81):
        # Note: Exact 80% split of unique IDs might result in slightly different token counts,
        # but the sentence split should be exact based on the code logic.
        # However, due to integer rounding in split index, it might vary slightly.
        # We just warn or assert closeness.
        print(
            "Warning: Split ratio deviates slightly from 0.80 due to integer rounding or distribution."
        )

    print("\nVerification Passed successfully.")


if __name__ == "__main__":
    generate_metadata()
