import pandas as pd
import numpy as np
import torch
import os
import sys

# Import classes and functions from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import process_data
from library.train_engine import run_fold
from library.inference import generate_submission


def main():
    print("--- Starting Skin Lesion Classification Demo ---")

    # 1. Setup Environment
    seed_everything(42)

    # Define temporary directories for the demo
    demo_base_dir = "./working/demo_run"
    demo_meta_dir = os.path.join(demo_base_dir, "metadata")
    demo_work_dir = os.path.join(demo_base_dir, "working")
    demo_sub_path = os.path.join(demo_base_dir, "submission.csv")

    os.makedirs(demo_meta_dir, exist_ok=True)
    os.makedirs(demo_work_dir, exist_ok=True)

    # 2. Prepare Data Subset for Speed
    # We load the original metadata and sample a small balanced subset.
    print("1. Preparing data subset...")

    orig_train = pd.read_csv("./metadata/train.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Select 20 benign and 20 malignant samples to ensure both classes exist for AUC
    # Note: Real dataset is highly imbalanced, so we explicitly pick positives.
    df_pos = orig_train[orig_train["target"] == 1].head(20)
    df_neg = orig_train[orig_train["target"] == 0].head(20)

    # Combine and shuffle
    subset_train_full = (
        pd.concat([df_pos, df_neg])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Assign unique patient IDs to ensure StratifiedGroupKFold behaves like StratifiedKFold
    # (prevents grouping all samples into one fold due to small sample size)
    subset_train_full["patient_id"] = [
        f"demo_pid_{i}" for i in range(len(subset_train_full))
    ]

    # Split into 'train' and 'val' files (process_data will concatenate them back, but we mimic input structure)
    split_idx = len(subset_train_full) // 2
    subset_train = subset_train_full.iloc[:split_idx]
    subset_val = subset_train_full.iloc[split_idx:]

    # Subset test data
    subset_test = orig_test.head(10).copy()

    # Save subset metadata
    path_train = os.path.join(demo_meta_dir, "train.csv")
    path_val = os.path.join(demo_meta_dir, "val.csv")
    path_test = os.path.join(demo_meta_dir, "test.csv")

    subset_train.to_csv(path_train, index=False)
    subset_val.to_csv(path_val, index=False)
    subset_test.to_csv(path_test, index=False)

    print(
        f"   Created subset: {len(subset_train_full)} train samples, {len(subset_test)} test samples."
    )

    # 3. Override Configuration
    # We modify the Config class attributes directly to control the pipeline
    print("2. Configuring pipeline parameters...")

    Config.TRAIN_METADATA_PATH = path_train
    Config.VAL_METADATA_PATH = path_val
    Config.TEST_METADATA_PATH = path_test
    Config.WORKING_DIR = demo_work_dir
    Config.SUBMISSION_PATH = demo_sub_path

    # Optimization settings for demo speed
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_FOLDS = 2  # Minimum folds required for splitting
    Config.DEBUG = True

    # 4. Data Processing
    print("3. Running data processing...")
    # load_cached_data=False forces the pipeline to process our new subset
    df_train_proc, df_test_proc, feature_cols = process_data(load_cached_data=False)

    # Verification
    assert len(df_train_proc) == 40, "Processed training set size mismatch"
    assert len(df_test_proc) == 10, "Processed test set size mismatch"
    assert "fold" in df_train_proc.columns, "Fold column not created"
    assert len(feature_cols) > 0, "Tabular features not generated"
    print("   Data processing successful.")

    # 5. Training Loop
    print("4. Training model (Fold 0)...")
    # We only run Fold 0 to save time.
    best_auc = run_fold(fold=0, load_cached_data=True)

    print(f"   Training finished. Best Validation AUC: {best_auc:.4f}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(demo_work_dir, "fold_0_best.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    # 6. Inference
    print("5. Generating submission...")

    # IMPORTANT: The inference script averages predictions over Config.NUM_FOLDS models.
    # Since we only trained Fold 0, we temporarily set NUM_FOLDS to 1 so the average is correct.
    # (Otherwise, it would divide the sum of probabilities by 2).
    Config.NUM_FOLDS = 1

    generate_submission(load_cached_data=True)

    # 7. Final Verification
    if not os.path.exists(demo_sub_path):
        raise FileNotFoundError("Submission file was not generated.")

    sub_df = pd.read_csv(demo_sub_path)

    # Check format
    assert sub_df.shape == (
        10,
        2,
    ), f"Submission shape mismatch. Expected (10, 2), got {sub_df.shape}"
    assert list(sub_df.columns) == [
        "image_name",
        "target",
    ], "Submission columns incorrect"
    assert (
        sub_df["target"].min() >= 0.0 and sub_df["target"].max() <= 1.0
    ), "Probabilities out of bounds"

    print("\n--- Demo Completed Successfully ---")
    print(f"Submission saved to: {demo_sub_path}")
    print(sub_df.head())


if __name__ == "__main__":
    main()
