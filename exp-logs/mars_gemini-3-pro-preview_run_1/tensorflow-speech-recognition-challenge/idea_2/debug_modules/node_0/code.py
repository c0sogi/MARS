import os
import sys
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.dataset import SpeechCommandsDataset, get_dataloaders
from library.model import ConvNeXtAudio
from library.trainer import train_model


def run_demonstration():
    print("=== Starting Speech Command Recognition Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configure for Fast Execution
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config defaults for speed and isolation
    Config.subset_size = 50  # Only use 50 samples for train/val/test
    Config.epochs = 1  # Run only 1 epoch
    Config.batch_size = 10  # Small batch size
    Config.pretrained = False  # Disable downloading pretrained weights
    Config.num_workers = 0  # Use main process to avoid multiprocessing overhead in demo
    Config.mixup_alpha = 0.0  # Disable mixup for simple logic check

    # Redirect outputs to a demo directory
    Config.working_dir = "./working/demo_run/"
    Config.submission_dir = "./working/demo_submission/"
    Config.model_save_path = os.path.join(Config.working_dir, "demo_best_model.pth")
    Config.submission_path = os.path.join(Config.submission_dir, "demo_submission.csv")

    # Create necessary directories
    Config.setup()
    print("    Configuration updated. Output directories created.")

    # -------------------------------------------------------------------------
    # 2. Verify Dataset Logic
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Dataset and Audio Processing...")

    # Load a small slice of metadata manually
    if not os.path.exists(Config.train_metadata_path):
        raise FileNotFoundError(f"Metadata not found at {Config.train_metadata_path}")

    df_train_demo = pd.read_csv(Config.train_metadata_path).head(10)

    # Instantiate Dataset
    dataset = SpeechCommandsDataset(df_train_demo, phase="train", config=Config)

    # Fetch one sample
    spec, label = dataset[0]

    print(f"    Sample Spectrogram Shape: {spec.shape}")
    print(f"    Sample Label ID: {label}")

    # Assertions
    assert spec.ndim == 3, f"Expected 3D tensor (C, F, T), got {spec.ndim}"
    assert spec.shape[0] == 1, f"Expected 1 channel, got {spec.shape[0]}"
    assert (
        spec.shape[1] == Config.n_mels
    ), f"Expected {Config.n_mels} mel bins, got {spec.shape[1]}"
    assert isinstance(label, torch.Tensor), "Label should be a torch Tensor"
    print("    Dataset logic verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtAudio(pretrained=False).to(device)
    model.eval()

    # Create dummy input: (Batch, Channel, Freq, Time)
    # Time dimension is approx sample_rate / hop_length. 16000/512 ~ 32
    dummy_input = torch.randn(2, 1, Config.n_mels, 32).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        2,
        Config.num_classes,
    ), f"Expected output shape (2, {Config.num_classes}), got {output.shape}"
    print("    Model architecture verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Execute Training Pipeline
    # -------------------------------------------------------------------------
    print("\n[4] Executing Training Pipeline (Train -> Val -> Test)...")

    # Run the trainer with the subset_size
    # We set load_cached_data=False to force fresh loading from metadata
    train_model(load_cached_data=False, subset_size=Config.subset_size, patience=1)

    print("    Training pipeline execution complete.")

    # -------------------------------------------------------------------------
    # 5. Validate Submission Output
    # -------------------------------------------------------------------------
    print("\n[5] Validating Submission File...")

    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    df_sub = pd.read_csv(Config.submission_path)
    print(f"    Loaded submission with {len(df_sub)} rows.")
    print(f"    Columns: {df_sub.columns.tolist()}")

    # Assertions
    assert (
        "fname" in df_sub.columns and "label" in df_sub.columns
    ), "Submission file missing required columns."

    # Since we subsetted the test data to Config.subset_size, the submission should match
    assert (
        len(df_sub) == Config.subset_size
    ), f"Expected {Config.subset_size} predictions, found {len(df_sub)}."

    # Check if labels are valid
    valid_labels = set(Config.labels)
    pred_labels = set(df_sub["label"].unique())
    invalid_preds = pred_labels - valid_labels
    assert not invalid_preds, f"Found invalid labels in submission: {invalid_preds}"

    print("    Submission file validated successfully.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility for the demo script itself
    torch.manual_seed(42)
    np.random.seed(42)

    run_demonstration()
