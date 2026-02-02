import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Append current directory to system path to ensure local library imports work
sys.path.append(".")

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import WhaleEfficientNet
from library.losses import WeightedBCELoss
from library.trainer import Trainer


def run_demo():
    print("Starting Whale Detection Pipeline Demo...")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set a separate working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 12  # Small number of samples
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.MODEL_NAME = "tf_efficientnet_b0"  # Smaller model for speed

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Model: {Config.MODEL_NAME}")

    # ---------------------------------------------------------
    # 2. Data Loading Demo
    # ---------------------------------------------------------
    print("\n[2] initializing DataLoaders...")

    # This will process audio files into spectrograms and cache them
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force processing to demonstrate pipeline
        debug=Config.DEBUG,
    )

    # Fetch a batch to verify shapes
    images, labels, clips = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")  # Expected: (Batch, 1, Freq, Time)
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions
    assert images.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert images.shape[1] == 1, "Channel dimension should be 1 (grayscale spectrogram)"
    assert len(labels) == Config.BATCH_SIZE, "Label count mismatch"
    assert len(clips) == Config.BATCH_SIZE, "Clip name count mismatch"
    print("Data Loading verification passed.")

    # ---------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # ---------------------------------------------------------
    print("\n[3] Instantiating Model...")

    device = Config.DEVICE
    model = WhaleEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # No need to download weights for logic check
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Dummy forward pass
    dummy_input = images.to(device)
    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"
    print("Model verification passed.")

    # ---------------------------------------------------------
    # 4. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Loss Function...")

    # Instantiate WeightedBCELoss
    # Using arbitrary pos_weight for demo
    criterion = WeightedBCELoss(pos_weight_value=1.0, device=device)

    dummy_targets = labels.to(device).view(-1, 1)
    loss = criterion(outputs, dummy_targets)

    print(f"Calculated Loss: {loss.item()}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("Loss function verification passed.")

    # ---------------------------------------------------------
    # 5. Trainer Execution (Fit)
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    trainer = Trainer(model, optimizer, device=device)

    # Run fit
    trainer.fit(
        train_loader, val_loader, epochs=Config.EPOCHS, checkpoint_name="demo_best.pth"
    )

    # Verify Checkpoint
    checkpoint_path = os.path.join(Config.WORKING_DIR, "demo_best.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
    print(f"Training complete. Checkpoint saved to {checkpoint_path}")

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    print("\n[6] Generating Submission...")

    preds = trainer.generate_submission(test_loader, output_file=Config.SUBMISSION_FILE)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission shape: {df_sub.shape}")
    print(f"First few rows:\n{df_sub.head()}")

    # Assertions
    assert (
        "clip" in df_sub.columns and "probability" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission file is empty."
    # Check if probabilities are within [0, 1] (sigmoid output)
    assert (
        df_sub["probability"].min() >= 0.0 and df_sub["probability"].max() <= 1.0
    ), "Probabilities out of range."

    print("Submission verification passed.")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
