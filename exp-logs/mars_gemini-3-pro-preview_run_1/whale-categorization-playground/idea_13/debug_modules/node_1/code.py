import sys
import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders
from library.models import get_model
from library.engine import train_one_epoch, validate, inference
from library.pseudo_labeling import generate_pseudo_labels


def run_demo():
    print("=== Starting Whale Identification Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides
    # -------------------------------------------------------------------------
    # We modify the Config class directly to set up a fast, lightweight run.
    print("\n[1] Configuring environment for demo...")

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 32  # Small subset for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.WORKING_DIR = "./working/demo_execution"

    # Re-run setup to create the new working directories
    Config.setup()

    # Set reproducible seeds
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Initializing DataLoaders...")

    # load_cached_data=False ensures we don't rely on stale cache files
    train_loader, val_loader, test_loader, le = get_loaders(load_cached_data=False)

    print(f"    Classes detected: {len(le.classes_)}")

    # Verify Train Batch
    images, labels, names = next(iter(train_loader))
    print(f"    Train Batch Images: {images.shape}")
    print(f"    Train Batch Labels: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Image tensor shape mismatch"
    assert labels.shape == (Config.BATCH_SIZE,), "Label tensor shape mismatch"
    assert len(names) == Config.BATCH_SIZE, "Image names list length mismatch"

    # -------------------------------------------------------------------------
    # 3. Model Instantiation
    # -------------------------------------------------------------------------
    print("\n[3] Instantiating Model...")

    # Use resnet18 for speed, pretrained=False to avoid download/network issues
    model = get_model("resnet18", num_classes=len(le.classes_), pretrained=False)
    model.to(device)

    # Test Forward Pass (Training Mode - with labels)
    # ArcFace head requires labels to compute margin loss
    logits = model(images.to(device), labels.to(device))
    assert logits.shape == (
        Config.BATCH_SIZE,
        len(le.classes_),
    ), f"Logits shape mismatch: expected {(Config.BATCH_SIZE, len(le.classes_))}, got {logits.shape}"

    # Test Forward Pass (Inference Mode - no labels)
    # Should return scaled cosine similarities
    feats = model(images.to(device))
    assert feats.shape == (
        Config.BATCH_SIZE,
        len(le.classes_),
    ), "Inference output shape mismatch"

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Run one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, device)

    print(f"    Epoch Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # -------------------------------------------------------------------------
    # 5. Validation
    # -------------------------------------------------------------------------
    print("\n[5] Running Validation...")

    val_loss, map5_score = validate(model, val_loader, device, le)

    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    MAP@5 Score: {map5_score:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= map5_score <= 1.0, "MAP@5 score out of range"

    # -------------------------------------------------------------------------
    # 6. Inference
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    submission_df = inference(model, test_loader, device, le)

    print(f"    Submission shape: {submission_df.shape}")
    print(f"    First few rows:\n{submission_df.head()}")

    assert list(submission_df.columns) == ["Image", "Id"], "Submission columns mismatch"
    assert (
        len(submission_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} predictions, got {len(submission_df)}"

    # -------------------------------------------------------------------------
    # 7. Pseudo-Labeling
    # -------------------------------------------------------------------------
    print("\n[7] Testing Pseudo-Labeling Pipeline...")

    # Temporarily lower threshold to ensure we generate *some* pseudo-labels
    # for demonstration purposes (since the model is barely trained).
    original_threshold = Config.PSEUDO_LABEL_THRESHOLD
    Config.PSEUDO_LABEL_THRESHOLD = 0.0  # Accept everything (except new_whale)

    # Generate pseudo labels using the current model
    # We pass a list of models (ensemble size 1)
    augmented_train_df = generate_pseudo_labels(
        models=[model], device=device, label_encoder=le, load_cached_data=False
    )

    print(f"    Augmented Dataset Size: {len(augmented_train_df)}")

    # Verification
    # The augmented dataframe should contain original training data + pseudo labels
    # Since we set threshold to 0.0, we expect some additions unless all preds are 'new_whale'
    assert "Id" in augmented_train_df.columns, "Augmented dataframe missing 'Id' column"
    assert (
        "file_path" in augmented_train_df.columns
    ), "Augmented dataframe missing 'file_path'"

    # Restore threshold
    Config.PSEUDO_LABEL_THRESHOLD = original_threshold

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
