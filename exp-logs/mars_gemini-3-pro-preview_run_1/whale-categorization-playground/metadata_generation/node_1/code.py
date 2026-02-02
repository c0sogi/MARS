import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_IMG_DIR = "train"
TEST_IMG_DIR = "test"
RANDOM_STATE = 42


def run():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # ---------------------------------------------------------
    # 1. Process Training Data
    # ---------------------------------------------------------
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    df_train_raw = pd.read_csv(train_csv_path)

    # Verify file existence to ensure metadata only points to real files
    train_dir_abs = os.path.join(INPUT_DIR, TRAIN_IMG_DIR)
    if os.path.exists(train_dir_abs):
        existing_train_files = set(os.listdir(train_dir_abs))
        # Filter dataframe to only include existing images
        df_train_raw = df_train_raw[
            df_train_raw["Image"].isin(existing_train_files)
        ].copy()

    # Create relative file path column
    df_train_raw["file_path"] = df_train_raw["Image"].apply(
        lambda x: os.path.join(TRAIN_IMG_DIR, x)
    )

    # Splitting Strategy:
    # Many classes have only 1 sample. We cannot stratify these.
    # We will put singletons in TRAIN, and stratify the rest 80/20.
    # We increase the threshold to 5 to ensure that with a 0.2 split,
    # the validation set size is large enough to cover all classes (5 * 0.2 = 1).

    id_counts = df_train_raw["Id"].value_counts()
    singletons = id_counts[id_counts < 5].index
    multi_instances = id_counts[id_counts >= 5].index

    df_singletons = df_train_raw[df_train_raw["Id"].isin(singletons)]
    df_multi = df_train_raw[df_train_raw["Id"].isin(multi_instances)]

    # Perform stratified split on multi-instance classes
    if len(df_multi) > 0:
        train_multi, val_multi = train_test_split(
            df_multi, test_size=0.2, stratify=df_multi["Id"], random_state=RANDOM_STATE
        )
    else:
        train_multi = df_multi
        val_multi = pd.DataFrame(columns=df_train_raw.columns)

    # Combine singletons with the split training data
    df_train_final = pd.concat([train_multi, df_singletons], axis=0)

    # Shuffle the final training set
    df_train_final = df_train_final.sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    df_val_final = val_multi.reset_index(drop=True)

    # Save to metadata
    df_train_final.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    df_val_final.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    # We use sample_submission.csv to define the test set images
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    df_ss = pd.read_csv(sample_sub_path)

    # Verify file existence
    test_dir_abs = os.path.join(INPUT_DIR, TEST_IMG_DIR)
    if os.path.exists(test_dir_abs):
        existing_test_files = set(os.listdir(test_dir_abs))
        df_ss = df_ss[df_ss["Image"].isin(existing_test_files)].copy()

    df_ss["file_path"] = df_ss["Image"].apply(lambda x: os.path.join(TEST_IMG_DIR, x))

    # Select only relevant columns for metadata (Image, file_path)
    # Note: The Id in sample_submission is dummy data, so we don't include it as a label.
    df_test_final = df_ss[["Image", "file_path"]]

    df_test_final.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")

    # ---------------------------------------------------------
    # 3. Validation & Checks
    # ---------------------------------------------------------
    print("Running verification checks...")

    # Load datasets back
    d_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    d_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    d_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 3.1 Print Summary Statistics
    print(
        f"Train Set: {len(d_train)} samples, {d_train['Id'].nunique()} unique classes."
    )
    print(f"Val Set:   {len(d_val)} samples, {d_val['Id'].nunique()} unique classes.")
    print(f"Test Set:  {len(d_test)} samples.")

    # 3.2 Check File Paths
    for name, df in [("Train", d_train), ("Val", d_val), ("Test", d_test)]:
        if len(df) == 0:
            print(f"Warning: {name} set is empty.")
            continue

        # Select random sample
        n_check = min(1000, len(df))
        sample = df.sample(n=n_check, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(row["file_path"])

        ratio = missing_count / n_check
        print(f"[{name}] Missing file ratio: {ratio:.4f}")

        if ratio > 0.5:
            print(f"Sample missing files: {missing_examples}")
            raise FileNotFoundError(f"Error: Too many missing files in {name} dataset.")

    # 3.3 Verify Split Requirements
    # Assert no overlap
    train_imgs = set(d_train["Image"])
    val_imgs = set(d_val["Image"])
    assert (
        len(train_imgs.intersection(val_imgs)) == 0
    ), "Error: Overlap detected between Train and Validation sets."

    # Assert stratification logic
    # Since we forced singletons to train, the classes in Val must be a subset of classes in Train.
    val_classes = set(d_val["Id"].unique())
    train_classes = set(d_train["Id"].unique())

    if not val_classes.issubset(train_classes):
        diff = val_classes - train_classes
        raise AssertionError(
            f"Error: Validation set contains classes not present in Training set: {list(diff)[:5]}"
        )

    print("All verification checks passed successfully.")


if __name__ == "__main__":
    run()
