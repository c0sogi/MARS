import os
import shutil
import pandas as pd
import numpy as np
import logging

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, setup_logger
from library.trainer import Trainer
from library.predictor import Predictor


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Initializing demo configuration...")

    # Define paths for the demo execution
    demo_working_dir = "./working/demo_execution"
    demo_metadata_dir = "./working/demo_metadata"

    # Clean up any previous demo runs to ensure a fresh start
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    if os.path.exists(demo_metadata_dir):
        shutil.rmtree(demo_metadata_dir)

    os.makedirs(demo_metadata_dir, exist_ok=True)

    # Override Config parameters for the demo
    # We use a separate directory and reduce folds for speed
    Config.WORKING_DIR = demo_working_dir
    Config.SUBMISSION_DIR = os.path.join(demo_working_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.N_FOLDS = 2

    # Re-run setup to create the new working directories defined above
    Config.setup()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Data Subsetting (Optimization)
    # ==========================================
    print("Creating data subsets for rapid execution...")

    # Load the original metadata
    # We assume these files exist as per the problem description
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create small subsets to ensure the pipeline runs quickly
    # 60 train, 20 val, 20 test samples are sufficient to demonstrate functionality
    sub_train = orig_train.head(60)
    sub_val = orig_val.head(20)
    sub_test = orig_test.head(20)

    # Save the subsets to the demo metadata directory
    train_meta_path = os.path.join(demo_metadata_dir, "train.csv")
    val_meta_path = os.path.join(demo_metadata_dir, "val.csv")
    test_meta_path = os.path.join(demo_metadata_dir, "test.csv")

    sub_train.to_csv(train_meta_path, index=False)
    sub_val.to_csv(val_meta_path, index=False)
    sub_test.to_csv(test_meta_path, index=False)

    # Update Config to point to these new subset metadata files
    Config.TRAIN_METADATA_PATH = train_meta_path
    Config.VAL_METADATA_PATH = val_meta_path
    Config.TEST_METADATA_PATH = test_meta_path

    print(
        f"Subset sizes -> Train: {len(sub_train)}, Val: {len(sub_val)}, Test: {len(sub_test)}"
    )

    # ==========================================
    # 3. Pipeline Execution: Training
    # ==========================================
    print("\n--- Starting Training Phase ---")
    trainer = Trainer()

    # Run Cross-Validation in debug mode (limits to 2 folds)
    # This handles feature extraction, densification, and model training
    trainer.run_cross_validation(debug=True)

    # Verify that model artifacts were successfully saved
    print("Verifying model artifacts...")
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    expected_artifacts = [
        "pipeline_fold_0.pkl",
        "pipeline_fold_1.pkl",
        "label_encoder.pkl",
    ]

    for artifact in expected_artifacts:
        path = os.path.join(models_dir, artifact)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Critical artifact missing: {path}")

    print("All training artifacts verified.")

    # ==========================================
    # 4. Pipeline Execution: Inference
    # ==========================================
    print("\n--- Starting Inference Phase ---")
    predictor = Predictor()

    # Generate submission file using the trained ensemble
    predictor.generate_submission()

    # ==========================================
    # 5. Validation of Results
    # ==========================================
    print("\n--- Validating Submission ---")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # 1. Validate Row Count
    # Should match the number of test images in our subset
    expected_rows = len(sub_test)
    if len(df_sub) != expected_rows:
        raise AssertionError(
            f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
        )

    # 2. Validate 'id' column presence
    if "id" not in df_sub.columns:
        raise AssertionError("Submission file is missing the required 'id' column.")

    # 3. Validate Probability Range
    # Extract all columns except 'id'
    prob_cols = [c for c in df_sub.columns if c != "id"]
    probs = df_sub[prob_cols].values

    if probs.min() < 0 or probs.max() > 1:
        raise AssertionError(
            f"Probabilities out of range [0, 1]. Min: {probs.min()}, Max: {probs.max()}"
        )

    # 4. Validate IDs match input
    # The IDs in the submission should match the IDs in our test subset
    sub_ids = set(df_sub["id"].values)
    expected_ids = set(sub_test["id"].values)
    if sub_ids != expected_ids:
        raise AssertionError("Submission IDs do not match the test dataset IDs.")

    print("Submission validation passed successfully.")
    print(f"Output saved to: {Config.SUBMISSION_PATH}")
    print("\nDemo execution completed.")


if __name__ == "__main__":
    main()
