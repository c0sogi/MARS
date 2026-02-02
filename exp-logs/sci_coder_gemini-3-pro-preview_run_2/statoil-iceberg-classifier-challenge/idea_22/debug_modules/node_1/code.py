import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_data, make_dataloaders, make_test_dataloader
from library.model import CADPNet
from library.engine import train_fold, predict


def run_demo():
    print("Starting Iceberg Classification Demo...")

    # ==========================================
    # 1. SETUP & CONFIGURATION OVERRIDE
    # ==========================================
    # Modify Config for a fast demonstration run
    print("Configuring environment for demo...")
    Config.WORK_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = Config.WORK_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Reduce computational load for demo
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.NUM_FOLDS = 2  # Run only 2 folds instead of 5
    Config.BATCH_SIZE = 32  # Smaller batch size
    Config.PATIENCE = 2  # Short patience

    # Create directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. DATA LOADING & VERIFICATION
    # ==========================================
    print("\nLoading data...")
    train_data, test_data = get_data(load_cached_data=True)

    # Verify Data Structure
    print("Verifying data integrity...")
    assert "images" in train_data
    assert "labels" in train_data
    assert "angles" in train_data
    assert train_data["images"].shape[1:] == (
        75,
        75,
        2,
    ), f"Unexpected image shape: {train_data['images'].shape}"
    assert len(train_data["images"]) == len(train_data["labels"])
    print("Data verification passed.")

    # ==========================================
    # 3. MODEL ARCHITECTURE CHECK
    # ==========================================
    print("\nVerifying model architecture...")
    model = CADPNet().to(device)

    # Create dummy input: Batch of 4, 3 channels (Dataset creates 3rd channel), 75x75
    dummy_img = torch.randn(4, 3, 75, 75).to(device)
    dummy_angle = torch.randn(4).to(device)

    # Forward pass check
    with torch.no_grad():
        output = model(dummy_img, dummy_angle)

    # Check output shape (Batch, 1)
    assert output.shape == (
        4,
        1,
    ), f"Model output shape mismatch. Expected (4, 1), got {output.shape}"
    print("Model architecture verification passed.")

    # ==========================================
    # 4. TRAINING LOOP DEMONSTRATION
    # ==========================================
    fold_models = []

    # Loop through the defined number of folds (2 for demo)
    for fold in range(Config.NUM_FOLDS):
        print(f"\n--- Processing Fold {fold + 1}/{Config.NUM_FOLDS} ---")

        # Create dataloaders for this specific fold
        # This function also returns scaler_stats computed on the training split of this fold
        train_loader, val_loader, scaler_stats = make_dataloaders(
            train_data, fold_idx=fold, batch_size=Config.BATCH_SIZE
        )

        # Verify dataloader content
        x_batch, a_batch, y_batch = next(iter(train_loader))
        assert x_batch.shape == (
            Config.BATCH_SIZE,
            3,
            75,
            75,
        )  # Check 3 channels creation

        # Train the model for this fold
        # engine.train_fold handles the training loop, validation, and checkpointing
        trained_model = train_fold(fold, train_loader, val_loader, device)

        # Store model and stats for inference
        fold_models.append((trained_model, scaler_stats))

    # ==========================================
    # 5. INFERENCE & SUBMISSION
    # ==========================================
    print("\nGenerating predictions...")

    test_ids = test_data["ids"]
    ensemble_preds = []

    # Generate predictions from each fold's model
    for i, (model, stats) in enumerate(fold_models):
        print(f"Predicting with model from Fold {i+1}...")

        # Create test loader using the SAME scaling stats as the training fold
        test_loader = make_test_dataloader(
            test_data, scaler_stats=stats, batch_size=Config.BATCH_SIZE
        )

        # Predict
        preds = predict(model, test_loader, device)
        ensemble_preds.append(preds)

        # Verify prediction shape
        assert preds.shape == (len(test_ids), 1), "Prediction shape mismatch"

    # Average predictions (Ensemble)
    avg_preds = np.mean(ensemble_preds, axis=0).flatten()

    # Create Submission DataFrame
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    # Save to CSV
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verify File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Check first few lines
    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(df_check) == len(test_ids), "Submission row count mismatch."
    assert list(df_check.columns) == [
        "id",
        "is_iceberg",
    ], "Submission columns mismatch."

    print("\nDemo completed successfully!")
    print(f"Output available at: {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    run_demo()
