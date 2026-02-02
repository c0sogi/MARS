import os
import torch
import numpy as np
import pandas as pd
import sys

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.csk_resnet import CSKResNet18CRNN
from library.trainer import Trainer


def main():
    print("=== Starting Demonstration of Right Whale Detection Pipeline ===")

    # 1. Configure for Fast Demonstration
    # We modify the Config class attributes directly to use a small subset and run quickly.
    print("\n[Step 1] Configuring environment for debug run...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(
        f"Configuration set: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}, DEVICE={Config.DEVICE}"
    )

    # 2. Data Loading Verification
    print("\n[Step 2] Verifying Data Loading...")
    # Force reload to ensure we generate the debug cache
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch from training loader
    try:
        data_batch, label_batch = next(iter(train_loader))
        print(f"Successfully loaded a batch.")
        print(f"Data Shape: {data_batch.shape}")  # Expected: (B, 1, 128, T)
        print(f"Label Shape: {label_batch.shape}")  # Expected: (B,)

        # Assertions
        assert data_batch.shape[0] == Config.BATCH_SIZE, "Batch size mismatch in data."
        assert data_batch.shape[1] == 1, "Channel dimension should be 1 (Log-Mel)."
        assert (
            data_batch.shape[2] == Config.N_MELS
        ), f"Frequency dimension should be {Config.N_MELS}."
        assert (
            label_batch.shape[0] == Config.BATCH_SIZE
        ), "Batch size mismatch in labels."
        print("Data Loading assertions passed.")
    except Exception as e:
        print(f"Data Loading failed: {e}")
        raise e

    # 3. Model Architecture Verification
    print("\n[Step 3] Verifying Model Architecture (CSKResNet18CRNN)...")
    model = CSKResNet18CRNN().to(Config.DEVICE)
    model.eval()

    # Create a dummy input matching the data shape found above
    # Shape: (Batch, 1, F, T)
    dummy_input = torch.randn_like(data_batch).to(Config.DEVICE)

    try:
        with torch.no_grad():
            output = model(dummy_input)

        print(f"Model Output Shape: {output.shape}")

        # Assertions
        assert output.shape == (
            Config.BATCH_SIZE,
            1,
        ), f"Expected output shape ({Config.BATCH_SIZE}, 1), got {output.shape}"
        print("Model architecture assertions passed.")
    except Exception as e:
        print(f"Model verification failed: {e}")
        raise e

    # 4. Training Pipeline Verification
    print("\n[Step 4] Verifying Training Pipeline (Trainer)...")
    trainer = Trainer(device=torch.device(Config.DEVICE))

    # Run training for 1 epoch
    print("Starting training loop (1 epoch)...")
    try:
        trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)
        print("Training loop completed successfully.")
    except Exception as e:
        print(f"Training loop failed: {e}")
        raise e

    # 5. Prediction and Submission Verification
    print("\n[Step 5] Verifying Prediction and Submission...")
    try:
        clips, probs = trainer.predict(test_loader)

        print(f"Generated {len(clips)} predictions.")

        # Create submission dataframe
        submission_df = pd.DataFrame({"clip": clips, "probability": probs})

        # Assertions
        assert len(submission_df) > 0, "Submission DataFrame is empty."
        assert "clip" in submission_df.columns, "Missing 'clip' column."
        assert "probability" in submission_df.columns, "Missing 'probability' column."
        assert submission_df["probability"].min() >= 0.0, "Probabilities must be >= 0."
        assert submission_df["probability"].max() <= 1.0, "Probabilities must be <= 1."

        # Save to file (as per Config)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

        assert os.path.exists(
            Config.SUBMISSION_PATH
        ), "Submission file was not created."
        print(f"Submission file saved to {Config.SUBMISSION_PATH}")
        print("Prediction and Submission assertions passed.")

    except Exception as e:
        print(f"Prediction/Submission failed: {e}")
        raise e

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
