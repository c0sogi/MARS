import os
import sys
import torch
import pandas as pd
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, mixup_data
from library.dataset import get_dataloaders
from library.model import EfficientNetAudio
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Speech Commands Algorithm Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Fast Demonstration
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid execution...")

    # Override Config defaults to ensure the script runs quickly (within minutes)
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.DEBUG = True  # Enable debug mode
    Config.DEBUG_SUBSET_SIZE = 100  # Use only 100 samples

    # Use a specific directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")
    print("    Configuration complete.")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Initialize DataLoaders
    # load_cached_data=False forces the loader to process the raw metadata
    # and create a new balanced subset for this debug run.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, debug=Config.DEBUG, load_cached_data=False
    )

    # Fetch one batch to verify shapes
    try:
        images, labels = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Label Shape: {labels.shape}")

    # Assertions
    # Expected shape: (Batch_Size, 1, 128, ~100) -> 1 channel (spectrogram), 128 mels
    # Time dimension depends on hop_length and duration, usually ~100 for 1s clip
    assert images.dim() == 4, f"Expected 4D input tensor, got {images.dim()}"
    assert (
        images.shape[0] == Config.BATCH_SIZE
    ), f"Expected batch size {Config.BATCH_SIZE}, got {images.shape[0]}"
    assert images.shape[1] == 1, f"Expected 1 channel (mono), got {images.shape[1]}"
    assert labels.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch"

    print("    Data Loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Augmentation Verification (Mixup)
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Mixup Augmentation...")

    # Apply mixup on CPU for verification
    mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=0.4, device="cpu")

    assert mixed_images.shape == images.shape, "Mixed images shape mismatch"
    assert y_a.shape == labels.shape, "Target A shape mismatch"
    assert y_b.shape == labels.shape, "Target B shape mismatch"
    # Lambda should be a scalar (float or 0-d tensor)
    assert isinstance(lam, (float, int)) or (
        isinstance(lam, torch.Tensor) and lam.numel() == 1
    ), "Lambda should be scalar-like"

    print("    Mixup verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    # Instantiate model (pretrained=False for speed check and to avoid download)
    model = EfficientNetAudio(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = model(images)

    print(f"    Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"

    print("    Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Loop (Trainer)...")

    # Initialize Trainer
    # This will internally create a new model (with pretrained=True by default)
    trainer = Trainer(device=Config.DEVICE)

    # Run training for 1 epoch
    best_acc = trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS, patience=1)

    print(f"    Training complete. Best Validation Accuracy: {best_acc:.4f}")

    # Verify Checkpoint Creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint file not found at {checkpoint_path}"

    print("    Training loop and checkpointing verification passed.")

    # -------------------------------------------------------------------------
    # 6. Inference Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Inference and Submission...")

    # Run prediction
    trainer.predict(test_loader, output_path=Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission File Rows: {len(df_sub)}")
    print(f"    Submission Columns: {list(df_sub.columns)}")

    assert len(df_sub) > 0, "Submission file is empty."
    assert "fname" in df_sub.columns, "Missing 'fname' column."
    assert "label" in df_sub.columns, "Missing 'label' column."

    # Verify that labels are within the expected set
    valid_labels = set(Config.LABELS)
    predicted_labels = set(df_sub["label"].unique())
    assert predicted_labels.issubset(
        valid_labels
    ), "Submission contains invalid labels."

    print("    Inference verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
