import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import WORKING_DIR, SUBMISSION_PATH, SEED
from library.utils import seed_everything
from library.data_manager import DataManager
from library.model import PIRVNoiseModel
from library.trainer import Trainer


def main():
    # 1. Setup and Initialization
    print("=== Setting up environment ===")
    seed_everything(SEED)

    # Clean working directory to ensure a fresh run
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Training Demonstration
    print("\n=== Starting Training Demo ===")

    # Initialize DataManager for training
    # We use debug_size=500 to limit the dataset size for rapid demonstration
    print("Initializing DataManager (Train)...")
    dm_train = DataManager(mode="train", debug_size=500)
    train_loader, val_loader = dm_train.get_dataloaders()

    # Validate Dataloader output
    X_kin_batch, X_vis_batch, y_batch = next(iter(train_loader))
    print(
        f"Batch Shapes -> Kinematic: {X_kin_batch.shape}, Visual: {X_vis_batch.shape}, Target: {y_batch.shape}"
    )

    assert X_kin_batch.ndim == 2, "Kinematic input must be 2D"
    assert X_vis_batch.ndim == 2, "Visual input must be 2D"
    assert y_batch.ndim == 1, "Target must be 1D"

    # Initialize Model
    print("Initializing PIRVNoiseModel...")
    model = PIRVNoiseModel()

    # Verify Forward Pass
    model.eval()
    with torch.no_grad():
        dummy_out = model(X_kin_batch, X_vis_batch)
    assert dummy_out.shape == (
        X_kin_batch.size(0),
        1,
    ), f"Model output shape mismatch: {dummy_out.shape}"
    print("Model forward pass verification successful.")

    # Initialize Trainer and Train
    print("Initializing Trainer and running 1 epoch...")
    trainer = Trainer(model)
    trainer.train(train_loader, val_loader, epochs=1)

    # Verify Training Artifacts
    assert os.path.exists(trainer.best_model_path), "best_model.pth was not created"
    assert os.path.exists(
        trainer.best_threshold_path
    ), "best_threshold.npy was not created"
    print("Training artifacts verified.")

    # 3. Cache Cleanup
    # The DataManager saves processed data to fixed filenames (e.g., 'tracking_processed.parquet').
    # We must delete these before initializing DataManager('test') so it processes the test files
    # instead of loading the cached training data.
    print("\n=== Cleaning Cache for Inference ===")
    cache_files = ["tracking_processed.parquet", "visuals_processed.parquet"]
    for f in cache_files:
        path = os.path.join(WORKING_DIR, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed cached file: {f}")

    # 4. Inference Demonstration
    print("\n=== Starting Inference Demo ===")

    # Initialize DataManager for testing
    # This processes the full test set as required for submission
    print("Initializing DataManager (Test)...")
    dm_test = DataManager(mode="test")
    test_loader, meta_info = dm_test.get_dataloaders()

    print(f"Test Set Size: {len(meta_info)}")

    # Generate Predictions
    print("Running predictions...")
    preds, raw_logits = trainer.predict(test_loader)

    assert len(preds) == len(
        meta_info
    ), "Number of predictions does not match metadata rows"

    # 5. Submission Generation
    print("Generating submission file...")
    submission = pd.DataFrame({"contact_id": meta_info["contact_id"], "contact": preds})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    # Final Verification
    print(f"Submission saved to: {SUBMISSION_PATH}")
    saved_sub = pd.read_csv(SUBMISSION_PATH)

    # Check against expected submission size (from task description)
    expected_rows = 463243
    if len(saved_sub) != expected_rows:
        print(
            f"Note: Submission has {len(saved_sub)} rows. (Expected {expected_rows} for full test set)"
        )

    assert (
        "contact_id" in saved_sub.columns and "contact" in saved_sub.columns
    ), "Missing required columns"
    print("Submission format verified.")
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
