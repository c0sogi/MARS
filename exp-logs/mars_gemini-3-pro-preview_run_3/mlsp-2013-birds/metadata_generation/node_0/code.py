import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from skmultilearn.model_selection import iterative_train_test_split

    HAS_SKMULTILEARN = True
except ImportError:
    HAS_SKMULTILEARN = False

RANDOM_STATE = 42


def run():
    # 1. Setup Paths
    input_dir = "./input"
    metadata_dir = "./metadata"
    essential_dir = os.path.join(input_dir, "essential_data")
    os.makedirs(metadata_dir, exist_ok=True)

    # 2. Load Data
    # Load rec_id to filename mapping
    rec_map_path = os.path.join(essential_dir, "rec_id2filename.txt")
    # Try reading with header, if 'rec_id' not in columns, reload without header
    rec_map = pd.read_csv(rec_map_path)
    if "rec_id" not in rec_map.columns:
        rec_map = pd.read_csv(rec_map_path, header=None, names=["rec_id", "filename"])

    # Identify filename column
    fname_col = [
        c for c in rec_map.columns if "file" in c.lower() or "name" in c.lower()
    ]
    fname_col = fname_col[0] if fname_col else rec_map.columns[1]

    # Load CV folds
    folds_path = os.path.join(essential_dir, "CVfolds_2.txt")
    folds = pd.read_csv(folds_path)

    # Load Labels (variable length)
    labels_path = os.path.join(essential_dir, "rec_labels_test_hidden.txt")
    label_data = []
    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts:
                continue
            try:
                rid = int(parts[0])
                lbls = [x.strip() for x in parts[1:] if x.strip()]
                if "?" in lbls:
                    lbl_str = "?"
                else:
                    lbl_str = " ".join(lbls)
                label_data.append({"rec_id": rid, "labels": lbl_str})
            except ValueError:
                continue
    labels_df = pd.DataFrame(label_data)

    # 3. Merge Data
    df = rec_map.merge(folds, on="rec_id").merge(labels_df, on="rec_id")

    # 4. Construct Relative File Paths
    # Path format: essential_data/src_wavs/<filename>
    df["file_path"] = df[fname_col].apply(
        lambda x: os.path.join("essential_data", "src_wavs", x)
    )

    # 5. Split Data
    # Fold 0 = Train/Val source, Fold 1 = Test source
    train_orig = df[df["fold"] == 0].copy()
    test_df = df[df["fold"] == 1].copy()

    # Shuffle training data before splitting
    train_orig = train_orig.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    # Prepare for Stratification
    # Create binary matrix for labels (19 species, 0-18)
    num_classes = 19

    def get_ohe(label_str):
        vec = np.zeros(num_classes, dtype=int)
        if label_str == "?" or not label_str:
            return vec
        try:
            indices = [int(x) for x in label_str.split()]
            vec[indices] = 1
        except ValueError:
            pass
        return vec

    X = train_orig["rec_id"].values.reshape(-1, 1)
    y = np.stack(train_orig["labels"].apply(get_ohe).values)

    # Perform Split (80/20)
    # Use IterativeStratification if possible, else random
    split_done = False
    if HAS_SKMULTILEARN:
        try:
            # iterative_train_test_split returns X_train, y_train, X_test, y_test
            X_train, _, X_val, _ = iterative_train_test_split(X, y, test_size=0.2)

            train_ids = X_train.flatten()
            val_ids = X_val.flatten()

            train_df = train_orig[train_orig["rec_id"].isin(train_ids)].copy()
            val_df = train_orig[train_orig["rec_id"].isin(val_ids)].copy()
            split_done = True
            print("Used Iterative Stratification.")
        except Exception as e:
            print(f"Iterative Stratification failed: {e}")

    if not split_done:
        print("Falling back to random split.")
        train_df, val_df = train_test_split(
            train_orig, test_size=0.2, random_state=RANDOM_STATE
        )

    # 6. Save Metadata
    # Select columns to save
    cols = ["rec_id", "file_path", "labels", "fold"]

    train_df[cols].to_csv(os.path.join(metadata_dir, "train.csv"), index=False)
    val_df[cols].to_csv(os.path.join(metadata_dir, "val.csv"), index=False)
    test_df[cols].to_csv(os.path.join(metadata_dir, "test.csv"), index=False)

    # 7. Verification
    print("\n==== Dataset Summary ====")
    datasets = {"Train": train_df, "Validation": val_df, "Test": test_df}

    for name, d in datasets.items():
        print(f"\n{name} Set:")
        print(f"  Samples: {len(d)}")
        print(f"  Shape: {d.shape}")

        # Class distribution for labeled sets
        if name != "Test":
            lbl_matrix = np.stack(d["labels"].apply(get_ohe).values)
            class_counts = lbl_matrix.sum(axis=0)
            print(f"  Class Distribution (Species 0-18): {class_counts}")
            # Check for empty label sets (just for info)
            empty_labels = (lbl_matrix.sum(axis=1) == 0).sum()
            print(f"  Samples with no labels: {empty_labels}")

    # Check File Paths
    print("\n==== File Path Verification ====")
    all_data = pd.concat([train_df, val_df, test_df])
    # Sample up to 1000 paths
    sample_size = min(1000, len(all_data))
    sample_paths = all_data["file_path"].sample(
        n=sample_size, random_state=RANDOM_STATE
    )

    missing_files = []
    for p in sample_paths:
        full_path = os.path.join(input_dir, p)
        if not os.path.exists(full_path):
            missing_files.append(p)

    missing_ratio = len(missing_files) / sample_size
    print(
        f"Checked {sample_size} files. Missing: {len(missing_files)} (Ratio: {missing_ratio:.4f})"
    )

    if missing_ratio > 0.5:
        print("Sample of missing files:")
        for mp in missing_files[:5]:
            print(f"  {mp}")
        raise FileNotFoundError(
            f"Missing file ratio {missing_ratio:.2f} exceeds threshold of 0.5"
        )

    # Verify Split Integrity
    print("\n==== Split Verification ====")
    train_ids = set(train_df["rec_id"])
    val_ids = set(val_df["rec_id"])

    # Check disjointness
    intersection = train_ids.intersection(val_ids)
    if intersection:
        raise AssertionError(
            f"Train and Validation sets overlap! Overlapping IDs: {list(intersection)[:5]}"
        )

    # Check coverage of original fold 0
    orig_fold0_ids = set(train_orig["rec_id"])
    union_ids = train_ids.union(val_ids)
    if union_ids != orig_fold0_ids:
        missing_ids = orig_fold0_ids - union_ids
        extra_ids = union_ids - orig_fold0_ids
        raise AssertionError(
            f"Split does not cover original training set correctly. Missing: {len(missing_ids)}, Extra: {len(extra_ids)}"
        )

    print("Verification passed successfully.")


if __name__ == "__main__":
    run()
