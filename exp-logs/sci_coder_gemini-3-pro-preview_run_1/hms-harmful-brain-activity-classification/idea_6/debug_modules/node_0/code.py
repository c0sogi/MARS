import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import SymmetryAwareNet
from library.train import Trainer


def run_demo():
    # 1. Setup and Configuration
    print("--- 1. Initializing Configuration and Environment ---")
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Initialize working directories
    Config.init_directories()

    # Define demo parameters for speed
    DEBUG_SIZE = 20
    BATCH_SIZE = 4
    EPOCHS = 1
    DEVICE = Config.DEVICE

    print(f"Device: {DEVICE}")
    print(f"Debug Size: {DEBUG_SIZE}, Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")

    # 2. Data Loading Verification
    print("\n--- 2. Verifying Data Loading ---")
    # We use load_cached_data=False to skip writing cache files for this quick demo
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debugging to avoid multiprocessing overhead
        load_cached_data=False,
        debug_size=DEBUG_SIZE,
    )

    # Fetch one batch to verify shapes
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    left_eeg = batch["left_eeg"]
    right_eeg = batch["right_eeg"]
    spec = batch["spectrogram"]
    labels = batch["label"]

    print(f"Batch keys: {list(batch.keys())}")

    # Expected shapes based on Config
    # EEG: (Batch, Channels, Time) -> (4, 8, 5000)
    # Spectrogram: (Batch, 3, Height, Width) -> (4, 3, 512, 512)
    expected_eeg_shape = (
        BATCH_SIZE,
        len(Config.LEFT_HEMISPHERE_CHANNELS),
        Config.EEG_SEQ_LENGTH,
    )
    expected_spec_shape = (
        BATCH_SIZE,
        3,
        Config.SPEC_RESIZE_SIZE[0],
        Config.SPEC_RESIZE_SIZE[1],
    )
    expected_label_shape = (BATCH_SIZE, Config.NUM_CLASSES)

    # Assertions
    assert (
        left_eeg.shape == expected_eeg_shape
    ), f"Left EEG shape mismatch. Expected {expected_eeg_shape}, got {left_eeg.shape}"
    assert (
        right_eeg.shape == expected_eeg_shape
    ), f"Right EEG shape mismatch. Expected {expected_eeg_shape}, got {right_eeg.shape}"
    assert (
        spec.shape == expected_spec_shape
    ), f"Spectrogram shape mismatch. Expected {expected_spec_shape}, got {spec.shape}"
    assert (
        labels.shape == expected_label_shape
    ), f"Label shape mismatch. Expected {expected_label_shape}, got {labels.shape}"

    print("Data shapes verified successfully.")

    # 3. Model Verification
    print("\n--- 3. Verifying Model Architecture ---")
    model = SymmetryAwareNet()
    model.to(DEVICE)

    # Move batch to device
    left_eeg = left_eeg.to(DEVICE)
    right_eeg = right_eeg.to(DEVICE)
    spec = spec.to(DEVICE)

    # Forward pass
    logits = model(left_eeg, right_eeg, spec)

    # Check output shape
    assert logits.shape == (
        BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"

    print("Model forward pass successful. Logits shape verified.")

    # 4. Training Loop Demonstration
    print("\n--- 4. Demonstrating Training Loop ---")
    trainer = Trainer(model, DEVICE)

    # Run fit for 1 epoch
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    # Check if checkpoint was saved
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(
        checkpoint_path
    ), "Checkpoint file 'best_model.pth' was not created."

    print(f"Training loop complete. Checkpoint saved at {checkpoint_path}")

    # 5. Inference and Submission
    print("\n--- 5. Demonstrating Inference and Submission ---")

    # Predict on test set
    probs, eeg_ids = trainer.predict(test_loader)

    # Verify predictions
    assert probs.shape[1] == Config.NUM_CLASSES, "Prediction classes mismatch."
    assert len(probs) == len(eeg_ids), "Mismatch between predictions and IDs count."

    # Verify probability constraints (sum to 1)
    # Allow small floating point tolerance
    sums = np.sum(probs, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), "Predictions do not sum to 1.0"

    print(f"Inference successful. Generated {len(probs)} predictions.")

    # Create submission dataframe
    submission_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    submission_df = pd.DataFrame(probs, columns=submission_cols)
    submission_df.insert(0, "eeg_id", eeg_ids)

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = Config.SUBMISSION_FILE
    submission_df.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file was not created."
    print(f"Submission file generated at {submission_path}")
    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
