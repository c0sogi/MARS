import os
import sys
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_dataloaders
from library.model import ResNetBiGRUAttention
from library.trainer import Trainer
from library.utils import set_seed


def main():
    print("=== Starting Right Whale Detection Demonstration ===")

    # 1. Setup and Configuration Override
    # We override specific configurations to ensure the demo runs quickly within the time limit.
    # We utilize the existing cache in 'working/idea_2' to skip raw audio processing.
    print("[1/6] Configuring environment...")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Reduce epochs for demonstration speed
    Config.NUM_EPOCHS = 2
    # Use a larger batch size if GPU memory allows (A100 40GB is plenty) to speed up iteration
    Config.BATCH_SIZE = 128

    print(f"    Device: {Config.DEVICE}")
    print(f"    Epochs: {Config.NUM_EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading
    print("[2/6] Loading datasets...")
    # load_cached_data=True will look for .npy files in Config.WORKING_DIR
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Verify DataLoaders
    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches: {len(val_loader)}")
    print(f"    Test Batches: {len(test_loader)}")

    # Assert we have data
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Val loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Inspect one batch to verify shapes
    sample_batch = next(iter(train_loader))
    # Expected shape: (Batch, 1, F, T) -> (128, 1, 64, 4000) based on Config
    # F=64 (N_MELS), T=4000 (SR*DURATION = 2000*2)
    inputs = sample_batch["data"]
    labels = sample_batch["label"]

    print(f"    Input Batch Shape: {inputs.shape}")
    print(f"    Label Batch Shape: {labels.shape}")

    assert inputs.dim() == 4, f"Expected 4D input (B, C, F, T), got {inputs.dim()}"
    assert inputs.shape[1] == 1, f"Expected 1 channel, got {inputs.shape[1]}"
    assert (
        inputs.shape[2] == Config.N_MELS
    ), f"Expected {Config.N_MELS} mel bands, got {inputs.shape[2]}"

    # 3. Model Architecture Verification
    print("[3/6] Verifying model architecture...")
    model = ResNetBiGRUAttention().to(Config.DEVICE)

    # Create a dummy input tensor on the correct device
    dummy_input = torch.randn(2, 1, Config.N_MELS, Config.FIXED_NUM_SAMPLES).to(
        Config.DEVICE
    )

    # Perform forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")

    # Assert output shape is (Batch, 1) (Logits)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    # 4. Training
    print("[4/6] Starting training loop...")
    trainer = Trainer()

    # Fit the model
    # This will run for Config.NUM_EPOCHS (2)
    best_auc = trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    print(f"    Training finished. Best Val AUC: {best_auc:.4f}")

    # Assert that the model checkpoint was created
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"

    # 5. Inference
    print("[5/6] Generating predictions on test set...")
    trainer.predict(test_loader)

    # 6. Submission Verification
    print("[6/6] Verifying submission file...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_submission = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"    Submission Rows: {len(df_submission)}")
    print(f"    Columns: {list(df_submission.columns)}")

    # Verify columns
    assert "clip" in df_submission.columns, "Column 'clip' missing in submission."
    assert (
        "probability" in df_submission.columns
    ), "Column 'probability' missing in submission."

    # Verify row count matches test set size
    # Based on metadata generation logs: Test Set has 25149 samples
    expected_rows = 25149
    assert (
        len(df_submission) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_submission)}"

    # Verify probabilities are valid
    probs = df_submission["probability"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("\n=== Demonstration Complete: Success ===")


if __name__ == "__main__":
    main()
