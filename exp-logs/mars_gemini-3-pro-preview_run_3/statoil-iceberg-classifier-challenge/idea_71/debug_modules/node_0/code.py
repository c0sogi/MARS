import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
import library.config
import library.utils
import library.model
import library.data_loader
import library.train


def run_demo():
    print("=== Starting Iceberg Classification Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("1. Patching configuration for fast execution...")
    # The default config has 75 epochs and 5 folds. We reduce this for the demo.
    # We must patch the imported modules directly since they import these values.

    DEMO_EPOCHS = 1
    DEMO_FOLDS = 2
    DEMO_BATCH_SIZE = 16

    library.config.NUM_EPOCHS = DEMO_EPOCHS
    library.config.NUM_FOLDS = DEMO_FOLDS
    library.config.BATCH_SIZE = DEMO_BATCH_SIZE

    # Patch library.train
    library.train.NUM_EPOCHS = DEMO_EPOCHS
    library.train.NUM_FOLDS = DEMO_FOLDS
    library.train.BATCH_SIZE = DEMO_BATCH_SIZE

    # Patch library.data_loader
    library.data_loader.NUM_FOLDS = DEMO_FOLDS
    library.data_loader.BATCH_SIZE = DEMO_BATCH_SIZE

    print(f"   NUM_EPOCHS set to {library.train.NUM_EPOCHS}")
    print(f"   NUM_FOLDS set to {library.train.NUM_FOLDS}")
    print("   Configuration patched.\n")

    # ---------------------------------------------------------
    # 2. Data Loading and Processing Verification
    # ---------------------------------------------------------
    print("2. Verifying Data Loading and Processing...")

    # Load data using the data_loader module
    # This checks if cache exists, if not processes from scratch
    X_train, y_train, angle_train, X_test, ids_test, angle_test = (
        library.data_loader.process_data(load_cached_data=True)
    )

    # Verify Shapes
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   angle_train shape: {angle_train.shape}")
    print(f"   X_test shape: {X_test.shape}")

    # Assertions
    assert (
        len(X_train) == len(y_train) == len(angle_train)
    ), "Training data lengths mismatch"
    assert len(X_test) == len(ids_test) == len(angle_test), "Test data lengths mismatch"
    assert X_train.shape[1:] == (
        3,
        75,
        75,
    ), f"Unexpected image shape: {X_train.shape[1:]}"
    assert not np.isnan(X_train).any(), "X_train contains NaNs"

    # Check incidence angle processing (should contain floats and NaNs where 'na' was present)
    # Note: angle_train might have NaNs, which is expected before imputation
    nan_count = np.isnan(angle_train).sum()
    print(f"   Missing incidence angles in train: {nan_count}")

    print("   Data loading verified.\n")

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("3. Verifying Model Architecture (RTICNN)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = library.model.RTICNN().to(device)

    # Create dummy input
    batch_size = 4
    dummy_img = torch.randn(batch_size, 3, 75, 75).to(device)
    dummy_angle = torch.randn(batch_size).to(device)  # Angles are scalars per image

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_img, dummy_angle)

    print(f"   Model output shape: {output.shape}")

    # Assertions
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("   Model architecture verified.\n")

    # ---------------------------------------------------------
    # 4. Running Training Pipeline
    # ---------------------------------------------------------
    print("4. Executing Training Pipeline (run_training_process)...")
    print("   This will run the patched Cross-Validation loop.")

    # This function encapsulates the entire workflow:
    # CV Split -> Imputation -> Training -> Validation -> Inference -> Submission
    library.train.run_training_process()

    print("   Training pipeline execution complete.\n")

    # ---------------------------------------------------------
    # 5. Submission Verification
    # ---------------------------------------------------------
    print("5. Verifying Submission File...")

    submission_path = os.path.join(library.config.SUBMISSION_DIR, "submission.csv")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"   Submission file loaded. Shape: {df_sub.shape}")
    print(f"   Columns: {list(df_sub.columns)}")
    print(f"   First 3 rows:\n{df_sub.head(3)}")

    # Assertions
    assert (
        df_sub.shape[0] == 321
    ), f"Expected 321 rows (test set size), got {df_sub.shape[0]}"
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Missing required columns"
    assert (
        df_sub["is_iceberg"].min() >= 0 and df_sub["is_iceberg"].max() <= 1
    ), "Probabilities out of range [0, 1]"

    print("   Submission verified.\n")
    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    library.utils.set_seed(42)

    # Run the demonstration
    run_demo()
