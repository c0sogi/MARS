import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config, set_seed
from library.utils import map_prediction_to_label
from library.audio_transforms import DualChannelProcessor
from library.dataset import get_dataloaders, SpeechCommandDataset
from library.model import DualResEfficientNet
from library.trainer import Trainer
from library.inference import generate_submission


def run_demo():
    print("=== Starting Demonstration of Speech Command Recognition Pipeline ===")

    # 1. Setup and Configuration Override for Speed
    print("\n[1] Configuring environment for rapid demonstration...")
    set_seed(42)

    # Override Config constants to ensure the demo runs quickly
    # The original BATCH_SIZE is 128. We reduce it to ensure we get batches even with small subsets.
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2

    # We will use a small subset of data for training/validation
    DEMO_SUBSET_SIZE = 50

    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Subset Size: {DEMO_SUBSET_SIZE}")
    print(f"    Device: {Config.DEVICE}")

    # 2. Verify Utility Logic
    print("\n[2] Verifying Utility Functions...")
    # Test label mapping logic
    # 'bed' is an auxiliary label -> should map to 'unknown'
    # 'yes' is a target label -> should map to 'yes'
    # 'silence' -> 'silence'

    lbl_bed = Config.map_prediction_to_submission("bed")
    assert lbl_bed == "unknown", f"Expected 'unknown', got {lbl_bed}"

    lbl_yes = Config.map_prediction_to_submission("yes")
    assert lbl_yes == "yes", f"Expected 'yes', got {lbl_yes}"

    print("    Label mapping logic verified.")

    # 3. Verify Audio Processing (DualChannelProcessor)
    print("\n[3] Verifying Dual-Channel Audio Processor...")
    # Initialize processor (load_cached_data=False to test raw processing logic)
    processor = DualChannelProcessor(load_cached_data=False)

    # Create a dummy waveform: 1 second of white noise at 16kHz
    dummy_waveform = torch.randn(1, 16000)

    # Process
    spec = processor(dummy_waveform, mode="train", label="unknown")

    # Expected output: (2, 128, 101)
    # 2 channels (Freq + Time), 128 Mels, 101 Time frames (16000/160 + 1)
    print(f"    Input shape: {dummy_waveform.shape}")
    print(f"    Output spectrogram shape: {spec.shape}")

    assert spec.dim() == 3, "Output should be 3D (C, F, T)"
    assert spec.shape[0] == 2, "Output should have 2 channels"
    assert spec.shape[1] == 128, "Output should have 128 Mel bands"
    # Time dimension depends on hop length (160) and padding logic, usually 101 for 1 sec
    assert spec.shape[2] == 101, f"Expected 101 time frames, got {spec.shape[2]}"

    print("    Processor output shape verified.")

    # 4. Verify Data Loading and Collation
    print("\n[4] Verifying Data Loading and Mixup...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,  # Use cache if available for speed
        debug_subset_size=DEMO_SUBSET_SIZE,
    )

    # Fetch one batch
    inputs, targets = next(iter(train_loader))

    print(f"    Batch inputs shape: {inputs.shape}")
    print(f"    Batch targets shape: {targets.shape}")

    # Inputs: (B, 2, 128, 101)
    assert inputs.shape == (Config.BATCH_SIZE, 2, 128, 101)
    # Targets: (B, NumClasses) because MixupCollator converts to One-Hot
    assert targets.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)
    assert targets.dim() == 2, "Targets should be one-hot encoded (2D)"

    print("    DataLoader and MixupCollator verified.")

    # 5. Verify Model Architecture
    print("\n[5] Verifying DualResEfficientNet Model...")
    model = DualResEfficientNet(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    with torch.no_grad():
        # Move inputs to device
        inputs_dev = inputs.to(Config.DEVICE)
        outputs = model(inputs_dev)

    print(f"    Model output shape: {outputs.shape}")
    assert outputs.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)

    print("    Model forward pass verified.")

    # 6. Verify Training Loop (Trainer)
    print("\n[6] Executing Training Loop (1 Epoch, Subset)...")
    # Initialize Trainer with the subset
    # Note: Trainer internally calls get_dataloaders using Config.BATCH_SIZE, which we updated.
    trainer = Trainer(load_cached_data=True, debug_subset_size=DEMO_SUBSET_SIZE)

    # Run fit
    # We use the modified Config.EPOCHS = 1
    trainer.fit(epochs=Config.EPOCHS)

    # Check if model was saved
    assert os.path.exists(trainer.best_model_path), "Best model file was not created."
    print(f"    Training complete. Model saved to {trainer.best_model_path}")

    # 7. Verify Inference
    print("\n[7] Executing Inference on Test Set...")
    # Inference runs on the full test set (as per library code design).
    # On GPU, inference for ~6500 files is very fast.

    generate_submission(load_cached_data=True, batch_size=Config.BATCH_SIZE)

    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file not found."

    # Validate submission format
    df_sub = pd.read_csv(submission_path)
    print(f"    Submission generated with {len(df_sub)} rows.")
    print(f"    First few rows:\n{df_sub.head()}")

    assert "fname" in df_sub.columns and "label" in df_sub.columns
    assert len(df_sub) > 0

    # Check that labels are valid (12 classes)
    valid_labels = set(
        [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
            "silence",
            "unknown",
        ]
    )
    unique_preds = set(df_sub["label"].unique())
    invalid_preds = unique_preds - valid_labels
    assert not invalid_preds, f"Found invalid labels in submission: {invalid_preds}"

    print("    Submission format verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
