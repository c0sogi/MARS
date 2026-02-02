import os
import torch
import pandas as pd
import numpy as np
import logging
import shutil

# Import library modules
from library.config import Config
from library.utils import set_seed, init_logger, count_parameters
from library.dataset import SpeechCommandsDataset, get_dataloaders
from library.model import MultiResConvNeXtCRNN
from library.trainer import Trainer


def run_demonstration():
    print("=== Starting Demonstration ===")

    # 1. Setup
    # ---------------------------------------------------------
    set_seed(42)
    logger = init_logger()

    # Override Config for speed
    print("\n[Step 1] Configuring for fast demonstration...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid overhead for small data

    # Create temporary subset CSVs to speed up training demonstration
    # We will use the first 50 samples from train and 20 from val
    full_train_df = pd.read_csv(Config.TRAIN_CSV)
    full_val_df = pd.read_csv(Config.VAL_CSV)
    full_test_df = pd.read_csv(Config.TEST_CSV)

    demo_train_path = os.path.join(Config.WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(Config.WORKING_DIR, "demo_val.csv")
    demo_test_path = os.path.join(Config.WORKING_DIR, "demo_test.csv")

    # Ensure working dir exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Save subsets
    full_train_df.head(50).to_csv(demo_train_path, index=False)
    full_val_df.head(20).to_csv(demo_val_path, index=False)
    full_test_df.head(20).to_csv(demo_test_path, index=False)

    # Point Config to these temporary files
    Config.TRAIN_CSV = demo_train_path
    Config.VAL_CSV = demo_val_path
    Config.TEST_CSV = demo_test_path

    print(f"Created subset metadata: {demo_train_path}")

    # 2. Dataset Verification
    # ---------------------------------------------------------
    print("\n[Step 2] Verifying Dataset Logic...")
    # Load the subset dataframe
    df_subset = pd.read_csv(Config.TRAIN_CSV)
    dataset = SpeechCommandsDataset(df_subset, mode="train")

    # Fetch one sample
    spec, label_id = dataset[0]

    # Assertions
    # Shape should be (3, N_MELS, Time)
    # Time dimension depends on Config.N_SAMPLES / Config.HOP_LENGTH
    # 16000 / 160 = 100 frames.
    # Note: torchaudio MelSpectrogram might produce slightly different time frames depending on center/pad.
    # Usually 16000 samples with hop 160 gives 101 frames (centered).
    print(f"Spectrogram Shape: {spec.shape}")
    print(f"Label ID: {label_id}")

    assert spec.dim() == 3, "Spectrogram must have 3 dimensions (Channels, Freq, Time)"
    assert spec.shape[0] == 3, "Spectrogram must have 3 channels (Multi-Resolution)"
    assert spec.shape[1] == Config.N_MELS, f"Freq dimension must be {Config.N_MELS}"
    assert isinstance(label_id, int), "Label must be an integer"

    print("Dataset verification passed.")

    # 3. Model Verification
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")
    model = MultiResConvNeXtCRNN()
    model.to(Config.DEVICE)

    # Count parameters
    params = count_parameters(model)
    print(f"Model Parameters: {params:,}")

    # Create dummy input: (Batch, 3, N_MELS, Time)
    # Using the time dimension observed from dataset (e.g., 101)
    time_dim = spec.shape[2]
    dummy_input = torch.randn(2, 3, Config.N_MELS, time_dim).to(Config.DEVICE)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {output.shape}"

    print("Model verification passed.")

    # 4. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[Step 4] Running Training Loop (1 Epoch on subset)...")

    trainer = Trainer()

    # Run fit
    # This uses the modified Config paths pointing to the small CSVs
    trainer.fit(epochs=Config.EPOCHS)

    # Check if model file was created
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training."
    print(f"Model saved successfully at {Config.MODEL_SAVE_PATH}")

    # 5. Prediction Demonstration
    # ---------------------------------------------------------
    print("\n[Step 5] Running Prediction on Test Subset...")

    trainer.predict()

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    print("Sample submission rows:")
    print(sub_df.head())

    assert list(sub_df.columns) == ["fname", "label"], "Submission columns mismatch."
    assert (
        len(sub_df) == 20
    ), "Submission length mismatch (should be 20 for the subset)."

    print("Prediction verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
