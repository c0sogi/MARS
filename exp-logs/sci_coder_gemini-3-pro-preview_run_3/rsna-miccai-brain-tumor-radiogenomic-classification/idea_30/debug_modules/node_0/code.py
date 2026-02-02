import sys
import os
import pandas as pd
import torch
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import components from the provided library
from library.config import (
    TRAIN_META_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    IMG_SIZE,
    TOTAL_CHANNELS,
    CACHE_DIR,
)
from library.utils import seed_everything, get_device
from library.data_loader import process_patient
from library.model import HRVANet
from library.train import fit
from library.inference import predict_and_submit


def main():
    print("=== Glioblastoma Genetic Subtype Prediction Demo ===")

    # 1. Setup
    # Ensure reproducibility
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # 2. Verify Data Processing Logic
    # We manually process one patient to ensure the data pipeline works correctly
    # and to verify shapes before committing to the full training loop.
    print("\n[1/4] Verifying Data Processing Logic...")

    if os.path.exists(TRAIN_META_PATH):
        df_train = pd.read_parquet(TRAIN_META_PATH)
        if len(df_train) > 0:
            # Select the first patient
            sample_row = df_train.iloc[0]
            pid = sample_row["BraTS21ID"]
            print(f"Processing sample patient: {pid}")

            # Process the patient (Load DICOMs -> Resize -> Normalize -> Stack)
            X_sample, y_sample = process_patient(sample_row)

            # Verify shapes and types
            print(f"Generated Tensor Shape: {X_sample.shape}")
            print(f"Target Value: {y_sample}")

            expected_shape = (TOTAL_CHANNELS, IMG_SIZE, IMG_SIZE)
            assert (
                X_sample.shape == expected_shape
            ), f"Shape mismatch! Expected {expected_shape}, got {X_sample.shape}"
            assert isinstance(y_sample, float), "Target should be a float."

            print("Data processing logic verified successfully.")
        else:
            raise ValueError("Training metadata is empty.")
    else:
        raise FileNotFoundError(f"Training metadata not found at {TRAIN_META_PATH}")

    # 3. Verify Model Architecture
    # Instantiate the model and run a dummy forward pass to check compatibility.
    print("\n[2/4] Verifying Model Architecture...")
    model = HRVANet()
    model.to(device)
    model.eval()

    # Create a dummy batch: (Batch_Size=2, Channels=64, Height=320, Width=320)
    dummy_input = torch.randn(2, TOTAL_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Expecting (Batch_Size, 1) logits
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch! Expected (2, 1), got {output.shape}"
    print("Model architecture verified successfully.")

    # 4. Run Training Pipeline
    # Run the training loop for 1 epoch. This will:
    # - Check/Create cache for Train/Val sets (may take ~5-10 mins if not cached)
    # - Train the model
    # - Validate and save the best model to MODEL_SAVE_PATH
    print("\n[3/4] Running Training Pipeline (1 Epoch)...")

    # fit() handles dataloading and the training loop internally
    fit(epochs=1)

    # Verify model was saved
    if not os.path.exists(MODEL_SAVE_PATH):
        raise RuntimeError(
            f"Model file was not created at {MODEL_SAVE_PATH} after training."
        )

    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")

    # 5. Run Inference Pipeline
    # Load the saved model and generate predictions for the test set.
    print("\n[4/4] Running Inference Pipeline...")

    predict_and_submit()

    # Verify submission file
    if not os.path.exists(SUBMISSION_PATH):
        raise RuntimeError(f"Submission file was not created at {SUBMISSION_PATH}.")

    submission_df = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission generated with {len(submission_df)} rows.")

    # Basic validation of submission format
    assert "BraTS21ID" in submission_df.columns, "Submission missing 'BraTS21ID' column"
    assert (
        "MGMT_value" in submission_df.columns
    ), "Submission missing 'MGMT_value' column"
    assert len(submission_df) > 0, "Submission file is empty"

    print("Inference pipeline completed successfully.")
    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    main()
