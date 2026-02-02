import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import library components
from library.config import Config, set_seed
from library.audio_frontend import DifferentiableFrontend
from library.model import EfficientNetV2Speech
from library.trainer import Trainer
from library.utils import logger

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_subset_metadata(source_csv, dest_csv, n_samples=50, ensure_noise=False):
    """
    Creates a subset of the metadata CSV to speed up the demonstration.
    Ensures background noise is included for the training set.
    """
    df = pd.read_csv(source_csv)

    subset_df = pd.DataFrame()

    if ensure_noise and "is_background" in df.columns:
        # Separate noise and commands
        df_noise = df[df["is_background"] == True]
        df_commands = df[df["is_background"] == False]

        # Take all noise files (usually few) and sample commands
        subset_df = pd.concat(
            [
                df_noise,
                df_commands.sample(n=min(n_samples, len(df_commands)), random_state=42),
            ]
        )
    else:
        subset_df = df.sample(n=min(n_samples, len(df)), random_state=42)

    subset_df.to_csv(dest_csv, index=False)
    return len(subset_df)


def main():
    print("=== Starting Speech Command Recognition Demo ===")

    # 1. Setup Environment and Config
    set_seed(42)

    # Define a temporary working directory for this demo
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    print(f"Working directory: {demo_working_dir}")

    # 2. Create Data Subsets
    print("\n--- Creating Data Subsets for Speed ---")
    train_subset_path = os.path.join(demo_working_dir, "train_subset.csv")
    val_subset_path = os.path.join(demo_working_dir, "val_subset.csv")
    test_subset_path = os.path.join(demo_working_dir, "test_subset.csv")

    n_train = create_subset_metadata(
        Config.TRAIN_CSV, train_subset_path, n_samples=50, ensure_noise=True
    )
    n_val = create_subset_metadata(Config.VAL_CSV, val_subset_path, n_samples=20)
    n_test = create_subset_metadata(Config.TEST_CSV, test_subset_path, n_samples=20)

    print(f"Created subsets: Train={n_train}, Val={n_val}, Test={n_test}")

    # 3. Override Config
    print("\n--- Overriding Configuration ---")
    Config.WORKING_DIR = demo_working_dir
    Config.TRAIN_CSV = train_subset_path
    Config.VAL_CSV = val_subset_path
    Config.TEST_CSV = test_subset_path
    Config.SUBMISSION_DIR = os.path.join(demo_working_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Speed optimizations
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0

    # 4. Unit Test: Differentiable Frontend
    print("\n--- Testing Differentiable Frontend ---")
    frontend = DifferentiableFrontend().to(Config.DEVICE)
    frontend.eval()  # Disable masking for deterministic shape check

    # Create dummy waveform: (Batch=2, Time=16000)
    dummy_wav = torch.randn(2, Config.NUM_SAMPLES).to(Config.DEVICE)

    with torch.no_grad():
        spec = frontend(dummy_wav)

    print(f"Input Waveform Shape: {dummy_wav.shape}")
    print(f"Output Spectrogram Shape: {spec.shape}")

    # Assertions
    # Expected: (Batch, 1, N_MELS, TimeFrames)
    # TimeFrames = 1 + (16000 // 160) = 101
    assert spec.dim() == 4, "Spectrogram must be 4D (B, C, F, T)"
    assert spec.shape[1] == 1, "Channel dimension must be 1"
    assert spec.shape[2] == Config.N_MELS, f"Freq dimension must be {Config.N_MELS}"
    print("Frontend verification passed.")

    # 5. Unit Test: Model Architecture
    print("\n--- Testing EfficientNetV2Speech Model ---")
    model = EfficientNetV2Speech(pretrained=False).to(
        Config.DEVICE
    )  # No need to download weights for shape check
    model.eval()

    with torch.no_grad():
        logits = model(dummy_wav)

    print(f"Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Output shape must be (Batch, {Config.NUM_CLASSES})"
    print("Model verification passed.")

    # 6. Integration Test: Trainer
    print("\n--- Starting Trainer Integration ---")
    # Initialize Trainer (this loads datasets and model)
    trainer = Trainer()

    # Verify Dataset Loading
    print("Verifying Dataset Initialization...")
    # Check if noise bank is populated (since we ensured noise files in subset)
    if len(trainer.train_data.noise_bank) > 0:
        print(f"Noise bank loaded with {len(trainer.train_data.noise_bank)} clips.")
    else:
        print("Warning: Noise bank is empty. Check subset creation logic.")

    assert len(trainer.train_data) > 0, "Train data is empty"
    assert len(trainer.val_data) > 0, "Val data is empty"

    # Run Training Loop (1 Epoch)
    print("Executing Training Loop (1 Epoch)...")
    trainer.fit()

    # Verify Checkpoints
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created."
    print(f"Checkpoint verified at {best_model_path}")

    # 7. Inference and Submission
    print("\n--- Generating Submission ---")
    trainer.generate_submission()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print("Sample predictions:")
    print(df_sub.head())

    # Validate submission content
    assert df_sub.shape[1] == 2, "Submission must have 2 columns"
    assert list(df_sub.columns) == [
        "fname",
        "label",
    ], "Columns must be 'fname' and 'label'"
    assert (
        len(df_sub) == n_test
    ), f"Submission rows ({len(df_sub)}) match test subset size ({n_test})"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
