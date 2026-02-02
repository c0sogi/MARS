import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.preprocessing import EEGPreprocessor
from library.models import HybridEEGModel
from library.train import train_model
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Fast Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Create a separate directory for this demo run
    DEMO_DIR = "./working/demo_run"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_TRAIN_DATA = os.path.join(DEMO_DIR, "train_data.npy")
    Config.CACHE_TRAIN_TARGETS = os.path.join(DEMO_DIR, "train_targets.npy")
    Config.CACHE_VAL_DATA = os.path.join(DEMO_DIR, "val_data.npy")
    Config.CACHE_VAL_TARGETS = os.path.join(DEMO_DIR, "val_targets.npy")
    Config.MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Override Hyperparameters for speed
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 samples
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # ---------------------------------------------------------
    # 2. Verify Preprocessing Logic
    # ---------------------------------------------------------
    print("\n[2] Verifying Preprocessing Logic...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    sample_row = train_df.iloc[0]

    # Initialize Preprocessor
    preprocessor = EEGPreprocessor()

    # Process a single sample
    print(f"Processing sample eeg_id: {sample_row['eeg_id']}...")
    raw_feat, spec_feat, targets = preprocessor.process_sample(
        sample_row, is_test=False
    )

    # Assertions
    # Raw feature shape: (Time, Channels) -> (2500, 19) as per preprocessing.py
    expected_raw_shape = (Config.RAW_SEQUENCE_LENGTH, Config.N_CHANNELS)
    assert (
        raw_feat.shape == expected_raw_shape
    ), f"Raw feature shape mismatch. Expected {expected_raw_shape}, got {raw_feat.shape}"

    # Spec feature shape: (Channels, Freq, Time) -> (19, 64, 256) as per preprocessing.py
    expected_spec_shape = (Config.N_CHANNELS, Config.SPEC_HEIGHT, Config.SPEC_WIDTH)
    assert (
        spec_feat.shape == expected_spec_shape
    ), f"Spec feature shape mismatch. Expected {expected_spec_shape}, got {spec_feat.shape}"

    # Targets shape: (6,)
    assert targets.shape == (
        Config.N_CLASSES,
    ), f"Targets shape mismatch. Expected ({Config.N_CLASSES},), got {targets.shape}"

    print("Preprocessing verification passed: Shapes are correct.")

    # ---------------------------------------------------------
    # 3. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = HybridEEGModel(num_classes=Config.N_CLASSES, pretrained_spec=False)
    model.to(device)
    model.eval()

    # Create dummy input tensors
    # Model expects Raw: (Batch, Channels, Time) -> (B, 19, 2500)
    # Note: Preprocessor outputs (Time, Channels), DataLoader permutes it. We simulate DataLoader here.
    batch_size = 2
    dummy_raw = torch.randn(
        batch_size, Config.N_CHANNELS, Config.RAW_SEQUENCE_LENGTH
    ).to(device)

    # Model expects Spec: (Batch, Channels, Freq, Time) -> (B, 19, 64, 256)
    dummy_spec = torch.randn(
        batch_size, Config.N_CHANNELS, Config.SPEC_HEIGHT, Config.SPEC_WIDTH
    ).to(device)

    print(f"Forward pass with batch size {batch_size}...")
    with torch.no_grad():
        outputs = model(dummy_raw, dummy_spec)

    # Assertions
    assert outputs.shape == (
        batch_size,
        Config.N_CLASSES,
    ), f"Model output shape mismatch. Expected {(batch_size, Config.N_CLASSES)}, got {outputs.shape}"

    # Check Softmax (sum to 1)
    sums = outputs.sum(dim=1)
    assert torch.allclose(
        sums, torch.ones_like(sums), atol=1e-5
    ), "Model outputs do not sum to 1 (Softmax check failed)."

    print("Model verification passed: Forward pass successful.")

    # ---------------------------------------------------------
    # 4. Run Training Pipeline (Debug Mode)
    # ---------------------------------------------------------
    print("\n[4] Running Training Pipeline (Debug Mode)...")

    # This will load data (processing 50 samples), cache it, and train for 2 epochs
    best_loss = train_model(
        debug=True,
        load_cached=True,  # Will process first time, then load if run again
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    print(f"Training finished. Best Validation Loss: {best_loss:.4f}")

    # Verify model file was created
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Training failed to save model at {Config.MODEL_PATH}")

    print(f"Model saved successfully at {Config.MODEL_PATH}")

    # ---------------------------------------------------------
    # 5. Run Inference Pipeline (Debug Mode)
    # ---------------------------------------------------------
    print("\n[5] Running Inference Pipeline (Debug Mode)...")

    generate_submission(
        debug=True,
        load_cached=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        model_path=Config.MODEL_PATH,
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference failed to save submission at {Config.SUBMISSION_PATH}"
        )

    # Check content format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {sub_df.shape}")

    required_cols = ["eeg_id"] + Config.CLASS_NAMES
    assert (
        list(sub_df.columns) == required_cols
    ), f"Submission columns mismatch.\nExpected: {required_cols}\nGot: {list(sub_df.columns)}"

    # Check eeg_id count matches debug sample size
    # Note: In debug mode, get_test_dataloader loads Config.DEBUG_SAMPLE_SIZE samples.
    # The inference script also slices the metadata df to DEBUG_SAMPLE_SIZE.
    assert (
        len(sub_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(sub_df)}"

    print("Inference verification passed: Submission file is valid.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
