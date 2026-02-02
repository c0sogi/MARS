import os
import sys
import torch
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import set_seed, create_submission_file
from library.data_loader import get_dataloaders
from library.model import DilatedResidualBiGRU
from library.trainer import Trainer


def main():
    print("=== RNA Degradation Prediction: Library Usage Demo ===")

    # 1. Setup and Configuration Override for Speed
    # We override epochs to 2 to ensure the script completes quickly.
    print("\n[1] Configuring environment...")
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small demo

    # Ensure reproducibility
    set_seed(Config.SEED)
    print(f"    Device: {Config.DEVICE}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading
    print("\n[2] Loading and Preprocessing Data...")
    # We set load_cached_data=False to demonstrate the preprocessing logic from scratch.
    # In a real run, set this to True to save time.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # 3. Data Verification
    print("\n[3] Verifying Data Integrity...")
    try:
        # Fetch one batch from training loader
        inputs, targets, ids = next(iter(train_loader))

        print(f"    Input Batch Shape: {inputs.shape}")
        print(f"    Target Batch Shape: {targets.shape}")

        # Assertions
        # Inputs: (Batch, Seq_Len=107, Channels=14)
        assert (
            inputs.shape[1] == 107
        ), f"Expected sequence length 107, got {inputs.shape[1]}"
        assert inputs.shape[2] == 14, f"Expected 14 channels, got {inputs.shape[2]}"

        # Targets: (Batch, Seq_Scored=68, Targets=5)
        assert (
            targets.shape[1] == 68
        ), f"Expected scored length 68, got {targets.shape[1]}"
        assert targets.shape[2] == 5, f"Expected 5 targets, got {targets.shape[2]}"

        print("    Data shapes verified successfully.")

    except StopIteration:
        raise ValueError("Train loader is empty!")

    # 4. Model Initialization and Forward Pass Check
    print("\n[4] Initializing Model and Checking Forward Pass...")
    model = DilatedResidualBiGRU()
    model.to(Config.DEVICE)

    # Move batch to device
    inputs = inputs.to(Config.DEVICE)

    # Forward pass
    outputs = model(inputs)
    print(f"    Model Output Shape: {outputs.shape}")

    # Assertions
    # Output: (Batch, Seq_Len=107, Targets=5)
    # Note: Model outputs predictions for full length 107, even though targets are 68
    assert outputs.shape[0] == inputs.shape[0], "Batch size mismatch in output"
    assert (
        outputs.shape[1] == 107
    ), f"Expected output length 107, got {outputs.shape[1]}"
    assert outputs.shape[2] == 5, f"Expected 5 output targets, got {outputs.shape[2]}"

    print("    Model forward pass verified successfully.")

    # 5. Training Loop Demonstration
    print("\n[5] Starting Training Loop...")
    trainer = Trainer(model)

    # Fit the model
    # This handles training, validation, metric calculation, and checkpointing
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Check if model file was created
    if os.path.exists(Config.MODEL_PATH):
        print(f"    Model checkpoint saved at: {Config.MODEL_PATH}")
    else:
        # If model wasn't saved (e.g. no improvement), we force save for the demo
        print(
            "    No improvement triggered save. Saving current state manually for demo."
        )
        torch.save(model.state_dict(), Config.MODEL_PATH)

    # 6. Inference and Submission
    print("\n[6] Running Inference on Test Set...")
    # Predict returns numpy array
    preds = trainer.predict(test_loader)

    print(f"    Prediction Shape: {preds.shape}")

    # Verify Prediction Shape
    # Test set has 240 samples. Shape should be (240, 107, 5)
    assert preds.shape[0] == 240, f"Expected 240 test samples, got {preds.shape[0]}"
    assert (
        preds.shape[1] == 107
    ), f"Expected 107 sequence positions, got {preds.shape[1]}"
    assert preds.shape[2] == 5, f"Expected 5 targets, got {preds.shape[2]}"

    print("\n[7] Generating Submission File...")
    # Need list of IDs from test dataset
    test_ids = test_loader.dataset.ids

    create_submission_file(preds, test_ids)

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"    Submission file created at: {Config.SUBMISSION_PATH}")

        # Verify Submission Content
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission DataFrame Shape: {sub_df.shape}")
        print(f"    Submission Columns: {list(sub_df.columns)}")

        # Expected rows: 240 samples * 107 positions = 25680
        expected_rows = 240 * 107
        assert (
            len(sub_df) == expected_rows
        ), f"Expected {expected_rows} rows, got {len(sub_df)}"

        # Expected columns: id_seqpos + 5 targets
        expected_cols = ["id_seqpos"] + Config.TARGET_COLS
        assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

        print("    Submission file format verified successfully.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
