import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_data, get_dataloaders
from library.model import AppleDiseaseModel
from library.engine import run_fold
from library.stacking import StackingPipeline


def main():
    print(">>> Starting Apple Disease Detection Demo Script")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config attributes to run a fast, lightweight demo
    # We use a temporary working directory to avoid conflicts
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    Config.working_dir = demo_working_dir
    Config.submission_path = os.path.join(demo_working_dir, "submission.csv")

    # Enable debug mode to use a tiny subset of data (50 images)
    Config.debug = True

    # Reduce training complexity
    Config.num_folds = 2  # Only run 2 folds instead of 5
    Config.epochs = 1  # Only 1 epoch
    Config.batch_size = 4  # Small batch size for the small subset

    # SWA Settings: Trigger immediately to test SWA logic
    Config.use_swa = True
    Config.swa_start_epoch = 0

    # Replace heavy models with a single lightweight model (ResNet18)
    # This ensures the script finishes within the time limit even on CPU
    Config.models = [
        {
            "name": "resnet18",
            "image_size": 224,
            "use_gem": False,  # Standard pooling
            "dropout_rate": 0.0,
            "drop_path_rate": 0.0,
        }
    ]

    print(f"    Working Directory: {Config.working_dir}")
    print(f"    Debug Mode: {Config.debug}")
    print(f"    Models: {[m['name'] for m in Config.models]}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Preparation...")

    # Force regeneration of cache
    full_train_df, test_df = prepare_data(load_cached_data=False)

    print(f"    Train DataFrame Shape: {full_train_df.shape}")
    print(f"    Test DataFrame Shape: {test_df.shape}")

    # Validation
    assert not full_train_df.empty, "Training DataFrame is empty."
    assert not test_df.empty, "Test DataFrame is empty."
    assert "rust" in full_train_df.columns, "Target column 'rust' missing."
    assert "scab" in full_train_df.columns, "Target column 'scab' missing."

    # -------------------------------------------------------------------------
    # 3. DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        fold=0, image_size=224, batch_size=Config.batch_size, load_cached_data=True
    )

    # Fetch a single batch to verify shapes
    images, targets, ids = next(iter(train_loader))

    print(f"    Batch Images Shape: {images.shape}")
    print(f"    Batch Targets Shape: {targets.shape}")

    assert images.shape == (
        Config.batch_size,
        3,
        224,
        224,
    ), "Incorrect image batch shape."
    assert targets.shape == (Config.batch_size, 2), "Incorrect target batch shape."

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    # Instantiate model (pretrained=False to avoid download issues in strict envs,
    # though engine uses True. Here we just test the class logic).
    model = AppleDiseaseModel(
        model_name="resnet18", pretrained=False, num_classes=2, use_gem=False
    )
    model.eval()

    # Run forward pass
    with torch.no_grad():
        outputs = model(images)

    print(f"    Model Output Shape: {outputs.shape}")
    assert outputs.shape == (Config.batch_size, 2), "Model output shape mismatch."

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (2 Folds)...")

    # Iterate through the configured folds
    for fold in range(Config.num_folds):
        print(f"    >> Running Fold {fold}")
        # Run training for the single configured model
        run_fold(fold, Config.models[0])

        # Verify that model checkpoints were saved
        best_model_path = os.path.join(
            Config.working_dir, f"best_model_resnet18_fold_{fold}.pth"
        )
        swa_model_path = os.path.join(
            Config.working_dir, f"swa_model_resnet18_fold_{fold}.pth"
        )

        # In this demo config, SWA starts at epoch 0, so we expect SWA model.
        # Best model might not be saved if validation doesn't improve (unlikely in epoch 0 init),
        # but usually at least one save occurs.

        if not os.path.exists(swa_model_path) and not os.path.exists(best_model_path):
            raise FileNotFoundError(f"No model checkpoints found for Fold {fold}.")

    print("    Training loop completed.")

    # -------------------------------------------------------------------------
    # 6. Stacking Pipeline & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6] Running Stacking Pipeline...")

    pipeline = StackingPipeline()

    # This method orchestrates:
    # 1. Generating OOF predictions (using the trained models)
    # 2. Training the Meta-Learner (Logistic Regression)
    # 3. Generating Test predictions
    # 4. Creating the submission CSV
    pipeline.generate_submission()

    # -------------------------------------------------------------------------
    # 7. Final Output Verification
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Submission File...")

    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    submission_df = pd.read_csv(Config.submission_path)
    print("    Submission Head:")
    print(submission_df.head())

    # Verify Columns
    expected_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    if list(submission_df.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch.\nExpected: {expected_cols}\nGot: {list(submission_df.columns)}"
        )

    # Verify Rows (should match test set size)
    # Note: In debug mode, test set is also sampled to 50
    expected_rows = len(test_df)
    if len(submission_df) != expected_rows:
        raise AssertionError(
            f"Submission row count mismatch. Expected {expected_rows}, got {len(submission_df)}"
        )

    # Verify Probabilities sum to 1 (approx)
    prob_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    row_sums = submission_df[prob_cols].sum(axis=1)
    # Allow small float error
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise AssertionError("Probabilities do not sum to 1.0")

    print("\n>>> Demo Script Completed Successfully!")


if __name__ == "__main__":
    main()
