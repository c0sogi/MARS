import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import config
from library.utils import seed_everything, save_submission
from library.data import get_dataloaders, ForestDataset
from library.model import ParallelDCNResNet
from library.train import Trainer


def main():
    print("=== Starting Demonstration of Forest Cover Type Pipeline ===")

    # ------------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------------
    # Override config for a fast, isolated demo run
    demo_dir = "./working/demo_execution"

    # Clean up previous demo run if exists to ensure fresh execution
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Update paths
    config.paths.working_dir = demo_dir
    config.paths.train_X_path = os.path.join(demo_dir, "X_train.npy")
    config.paths.train_y_path = os.path.join(demo_dir, "y_train.npy")
    config.paths.val_X_path = os.path.join(demo_dir, "X_val.npy")
    config.paths.val_y_path = os.path.join(demo_dir, "y_val.npy")
    config.paths.test_X_path = os.path.join(demo_dir, "X_test.npy")
    config.paths.test_ids_path = os.path.join(demo_dir, "test_ids.npy")
    config.paths.model_save_path = os.path.join(demo_dir, "best_model.pth")
    config.paths.submission_path = os.path.join(demo_dir, "submission.csv")

    # Update training parameters for speed
    config.train.debug = True
    config.train.debug_sample_size = 1000  # Small subset
    config.train.batch_size = 32
    config.train.epochs = 1
    config.train.num_workers = 0  # Avoid multiprocessing overhead for small demo
    config.train.device = (
        "cpu"  # Force CPU for simple demo stability, or use cuda if available
    )
    if torch.cuda.is_available():
        config.train.device = "cuda"

    print(f"Configured for debug run in: {demo_dir}")
    print(f"Device: {config.train.device}")

    # Set seeds for reproducibility
    seed_everything(config.train.seed)

    # ------------------------------------------------------------------------
    # 2. Data Processing & Loading
    # ------------------------------------------------------------------------
    print("\n[Step 1] Processing Data and Creating DataLoaders...")

    # Force processing from scratch (load_cached_data=False) to test FeatureEngineer
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Verification: Check DataLoaders
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"
    assert len(test_loader) > 0, "Test loader is empty"
    assert (
        len(test_ids) == config.train.debug_sample_size
    ), f"Expected {config.train.debug_sample_size} test IDs, got {len(test_ids)}"

    # Verification: Check Batch Shape
    # Fetch one batch
    inputs, targets = next(iter(train_loader))
    print(f"Batch Shape - Inputs: {inputs.shape}, Targets: {targets.shape}")

    # Expected input dim is 59 (54 raw + 5 engineered)
    expected_dim = config.model.input_dim
    assert (
        inputs.shape[1] == expected_dim
    ), f"Expected input dim {expected_dim}, got {inputs.shape[1]}"
    assert (
        targets.shape[0] == config.train.batch_size
        or targets.shape[0] == config.train.debug_sample_size
    ), "Unexpected batch size"

    print("Data processing and loading verified successfully.")

    # ------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ------------------------------------------------------------------------
    print("\n[Step 2] Initializing Model and Verifying Forward Pass...")

    device = torch.device(config.train.device)
    model = ParallelDCNResNet().to(device)

    # Move sample batch to device
    inputs = inputs.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)

    print(f"Model Output Shape: {outputs.shape}")

    # Verification: Output shape should be (Batch_Size, Num_Classes)
    assert outputs.shape == (
        inputs.shape[0],
        config.model.num_classes,
    ), f"Expected output shape {(inputs.shape[0], config.model.num_classes)}, got {outputs.shape}"

    # Verification: No NaNs
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"

    print("Model architecture verified successfully.")

    # ------------------------------------------------------------------------
    # 4. Training Loop
    # ------------------------------------------------------------------------
    print("\n[Step 3] Running Training Loop (1 Epoch)...")

    trainer = Trainer(model, train_loader, val_loader, device)

    # Fit model (Runs for 1 epoch as configured)
    best_acc = trainer.fit()

    # Verification: Accuracy should be a float between 0 and 1
    print(f"Training finished. Best Validation Accuracy: {best_acc}")
    assert isinstance(best_acc, float), "Accuracy is not a float"
    assert 0.0 <= best_acc <= 1.0, "Accuracy out of bounds"

    # Check if model artifact was saved
    assert os.path.exists(config.paths.model_save_path), "Model checkpoint not found"

    print("Training loop verified successfully.")

    # ------------------------------------------------------------------------
    # 5. Inference & Submission
    # ------------------------------------------------------------------------
    print("\n[Step 4] Generating Predictions and Submission File...")

    predictions = trainer.predict(test_loader)

    # Verification: Prediction count matches test IDs
    assert len(predictions) == len(
        test_ids
    ), f"Prediction count ({len(predictions)}) does not match Test IDs ({len(test_ids)})"

    # Save submission
    save_submission(test_ids, predictions, config.paths.submission_path)

    # Verification: File existence and content
    assert os.path.exists(config.paths.submission_path), "Submission file not created"

    df_sub = pd.read_csv(config.paths.submission_path)
    print(f"Submission File Head:\n{df_sub.head()}")

    assert list(df_sub.columns) == ["Id", "Cover_Type"], "Incorrect submission columns"
    assert len(df_sub) == len(test_ids), "Submission row count mismatch"
    assert (
        df_sub["Cover_Type"].dtype == np.int64 or df_sub["Cover_Type"].dtype == np.int32
    ), "Cover_Type should be integer"

    print("Inference and submission verified successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
