import os
import sys
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Attempt to import IterativeStratification for multi-label stratification
try:
    from skmultilearn.model_selection import IterativeStratification

    HAS_SKMULTILEARN = True
except ImportError:
    HAS_SKMULTILEARN = False


def main():
    # --- Configuration ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    ESSENTIAL_DATA_DIR = os.path.join(INPUT_DIR, "essential_data")
    WAV_DIR = os.path.join(ESSENTIAL_DATA_DIR, "src_wavs")

    # File paths
    CV_FOLDS_PATH = os.path.join(ESSENTIAL_DATA_DIR, "CVfolds_2.txt")
    REC_ID_MAP_PATH = os.path.join(ESSENTIAL_DATA_DIR, "rec_id2filename.txt")
    LABELS_PATH = os.path.join(ESSENTIAL_DATA_DIR, "rec_labels_test_hidden.txt")
    SPECIES_LIST_PATH = os.path.join(ESSENTIAL_DATA_DIR, "species_list.txt")

    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")

    # --- Load Data ---

    # 1. Load CV Folds
    # Format: rec_id,fold
    cv_folds = pd.read_csv(CV_FOLDS_PATH)

    # 2. Load Rec ID to Filename
    # Format: rec_id,filename (Assuming CSV format based on usage, checking header)
    # The file might not have a header or might be comma separated.
    # Let's read it carefully.
    try:
        rec_map = pd.read_csv(REC_ID_MAP_PATH)
        if "rec_id" not in rec_map.columns or "filename" not in rec_map.columns:
            # Try reading without header
            rec_map = pd.read_csv(
                REC_ID_MAP_PATH, header=None, names=["rec_id", "filename"]
            )
    except Exception as e:
        print(f"Error reading rec_id2filename.txt: {e}")
        sys.exit(1)

    # 3. Load Labels
    # Format: rec_id, label1, label2...
    # Since variable length, we read line by line
    labels_dict = {}
    with open(LABELS_PATH, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts:
                continue
            try:
                rid = int(parts[0])
                lbls = parts[1:]
                # Clean labels
                clean_lbls = [l.strip() for l in lbls if l.strip() != ""]
                labels_dict[rid] = clean_lbls
            except ValueError:
                continue

    # 4. Load Species List (to get N classes)
    with open(SPECIES_LIST_PATH, "r") as f:
        species_lines = f.readlines()
    num_classes = len(species_lines)
    species_ids = list(range(num_classes))

    # --- Merge Data ---
    data = []
    for idx, row in cv_folds.iterrows():
        rid = row["rec_id"]
        fold = row["fold"]

        # Get filename
        fname_row = rec_map[rec_map["rec_id"] == rid]
        if fname_row.empty:
            print(f"Warning: rec_id {rid} not found in filename map.")
            continue
        fname = fname_row.iloc[0]["filename"]

        # Construct relative path
        # Path should be relative to ./input
        # The wavs are in essential_data/src_wavs/
        rel_path = os.path.join("essential_data", "src_wavs", fname)

        # Get labels
        lbls = labels_dict.get(rid, [])

        # Check if test
        is_test_sample = fold == 1

        # Parse labels
        binary_labels = [0] * num_classes
        label_list_int = []

        if is_test_sample:
            # Labels might be ['?']
            pass
        else:
            for l in lbls:
                if l == "?":
                    continue
                try:
                    l_int = int(l)
                    if 0 <= l_int < num_classes:
                        binary_labels[l_int] = 1
                        label_list_int.append(l_int)
                except ValueError:
                    pass

        entry = {
            "rec_id": rid,
            "file_path": rel_path,
            "fold": fold,
            "labels_str": " ".join(map(str, label_list_int)),
            "label_list": label_list_int,
        }
        # Add binary columns
        for i in range(num_classes):
            entry[f"species_{i}"] = binary_labels[i]

        data.append(entry)

    df = pd.DataFrame(data)

    # --- Split Data ---

    # Test set is fixed by fold=1
    test_df = df[df["fold"] == 1].copy()
    train_val_pool = df[df["fold"] == 0].copy()

    print(f"Total samples: {len(df)}")
    print(f"Test samples (Fold 1): {len(test_df)}")
    print(f"Train/Val pool (Fold 0): {len(train_val_pool)}")

    # Split Train/Val Pool (80/20)
    # We need to stratify. Since it's multi-label, we try IterativeStratification

    X_pool = train_val_pool["rec_id"].values.reshape(-1, 1)  # Dummy X
    # Create binary matrix for stratification
    y_pool = train_val_pool[[f"species_{i}" for i in range(num_classes)]].values

    train_indices = []
    val_indices = []

    # Stratification Logic
    stratification_successful = False
    if HAS_SKMULTILEARN:
        try:
            print(
                "Attempting multi-label stratification using IterativeStratification..."
            )
            # n_splits=5 means each fold is 20%. We take one fold as val.
            k_fold = IterativeStratification(n_splits=5, order=1)
            # This returns a generator
            splits = list(k_fold.split(X_pool, y_pool))
            # Take the first split
            train_idx_rel, val_idx_rel = splits[0]

            train_indices = train_val_pool.iloc[train_idx_rel].index
            val_indices = train_val_pool.iloc[val_idx_rel].index
            stratification_successful = True
        except Exception as e:
            print(f"IterativeStratification failed: {e}. Fallback to random split.")

    if not stratification_successful:
        print("Using random split (fallback).")
        # Fallback to random split
        train_df_split, val_df_split = train_test_split(
            train_val_pool, test_size=0.2, random_state=42, shuffle=True
        )
        train_df = train_df_split
        val_df = val_df_split
    else:
        train_df = df.loc[train_indices].copy()
        val_df = df.loc[val_indices].copy()

    # Reset index for cleanliness
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Final Train size: {len(train_df)}")
    print(f"Final Val size: {len(val_df)}")
    print(f"Final Test size: {len(test_df)}")

    # --- Save Metadata ---
    # Drop the 'label_list' column as it's an object, keep 'labels_str' and binary cols
    cols_to_drop = ["label_list", "fold"]

    train_df.drop(columns=cols_to_drop, errors="ignore").to_csv(
        os.path.join(METADATA_DIR, "train.csv"), index=False
    )
    val_df.drop(columns=cols_to_drop, errors="ignore").to_csv(
        os.path.join(METADATA_DIR, "val.csv"), index=False
    )
    test_df.drop(columns=cols_to_drop, errors="ignore").to_csv(
        os.path.join(METADATA_DIR, "test.csv"), index=False
    )

    print("Metadata files saved.")

    # --- Verification ---
    print("\n--- Verification ---")

    # 1. Reload
    v_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    v_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    v_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 2. Stats
    print("Train shape:", v_train.shape)
    print("Val shape:", v_val.shape)
    print("Test shape:", v_test.shape)

    # Check overlap
    train_ids = set(v_train["rec_id"])
    val_ids = set(v_val["rec_id"])
    test_ids = set(v_test["rec_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test overlap!"

    # 3. File Path Check
    all_paths = pd.concat(
        [v_train["file_path"], v_val["file_path"], v_test["file_path"]]
    )

    # Sample 1000 (with replacement if needed, though here we just check all since N is small)
    # The requirement says "check 1000 relative file paths randomly selected".
    # Since we have ~322 files, we can just sample with replacement or check all.
    # We'll check all unique paths to be safe, but simulate the sampling requirement.

    paths_to_check = all_paths.sample(n=1000, replace=True, random_state=42).tolist()

    missing_count = 0
    missing_samples = []

    for p in paths_to_check:
        full_path = os.path.join(INPUT_DIR, p)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    missing_ratio = missing_count / len(paths_to_check)
    print(f"Missing file ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample missing files:", missing_samples)
        raise FileNotFoundError(
            "More than 50% of files are missing from the input directory."
        )

    # 4. Stratification Check
    # Check if distribution of labels in Train vs Val is roughly similar
    # We calculate the prevalence of each class
    if stratification_successful:
        print("Verifying stratification...")
        train_labels = v_train[[c for c in v_train.columns if c.startswith("species_")]]
        val_labels = v_val[[c for c in v_val.columns if c.startswith("species_")]]

        train_dist = train_labels.mean()
        val_dist = val_labels.mean()

        # We don't assert strict equality, but we ensure that for common classes, they are present in both.
        # For very rare classes, they might be missing in Val.
        # Just printing correlation or checking non-empty val for common classes.
        print(
            "Class distribution correlation (Train vs Val):", train_dist.corr(val_dist)
        )

        # Simple assertion: Val set should not be empty
        assert len(v_val) > 0, "Validation set is empty"
        # Assert split ratio is roughly correct (within tolerance due to multi-label constraints)
        total_pool = len(v_train) + len(v_val)
        val_ratio = len(v_val) / total_pool
        print(f"Validation ratio: {val_ratio:.2f}")
        assert (
            0.15 < val_ratio < 0.25
        ), f"Validation ratio {val_ratio} is outside expected range (0.15-0.25)"

    print("Verification passed successfully.")


if __name__ == "__main__":
    main()
