import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Ensure the current directory is in the python path to import from library
sys.path.append(os.getcwd())

from library.config import Config
from library.data import get_loaders
from library.model import NBHACNN
from library.train import train_one_epoch, validate
from library.utils import set_seed


def main():
    # 1. Setup and Configuration
    print("Initializing Configuration...")
    # We use debug=True to use a subset of data and run fewer epochs for demonstration speed.
    config = Config(debug=True, epochs=1, batch_size=8)

    # Override working directory for this specific demo run to avoid cache conflicts
    config.working_dir = "./working/demo_run"
    config.cache_dir = config.working_dir
    config.checkpoint_dir = os.path.join(config.working_dir, "checkpoints")
    os.makedirs(config.working_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    set_seed(config.seed)
    print(f"Device: {config.device}")

    # 2. Data Loading Verification
    print("\nLoading Data (Fold 0)...")
    # This triggers cache generation if not present
    train_loader, val_loader, test_loader = get_loaders(config, fold_idx=0)

    # Verify Train Loader
    images, angles, targets = next(iter(train_loader))
    print(
        f"Train Batch - Images: {images.shape}, Angles: {angles.shape}, Targets: {targets.shape}"
    )

    # Assertions to ensure data integrity
    assert images.shape == (config.batch_size, 3, 75, 75), "Incorrect image batch shape"
    assert angles.shape == (config.batch_size,), "Incorrect angle batch shape"
    assert targets.shape == (config.batch_size, 1), "Incorrect target batch shape"
    assert not torch.isnan(images).any(), "NaN values found in images"

    # Verify Test Loader
    test_images, test_angles, test_ids = next(iter(test_loader))
    print(
        f"Test Batch  - Images: {test_images.shape}, Angles: {test_angles.shape}, IDs: {len(test_ids)}"
    )
    assert len(test_ids) == config.batch_size, "Incorrect test batch size"

    # 3. Model Initialization and Verification
    print("\nInitializing Model...")
    model = NBHACNN(config).to(config.device)

    # Dummy forward pass to verify architecture
    dummy_img = torch.randn(2, 3, 75, 75).to(config.device)
    dummy_ang = torch.randn(2).to(config.device)  # (Batch,)

    with torch.no_grad():
        output = model(dummy_img, dummy_ang)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), "Model output shape mismatch"

    # 4. Training Loop Demonstration
    print("\nStarting Training Demonstration...")
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    # Train for one epoch
    train_loss = train_one_epoch(
        train_loader, model, optimizer, criterion, config.device
    )
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")

    # Validate
    val_loss, val_metric = validate(val_loader, model, criterion, config.device)
    print(f"Epoch 1 Val Loss: {val_loss:.4f} | Val Log Loss: {val_metric:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(config.checkpoint_dir, "demo_model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # 5. Inference and Submission Generation
    print("\nGenerating Predictions on Test Set...")
    model.eval()

    predictions = []
    ids_list = []

    with torch.no_grad():
        for images, angles, ids in test_loader:
            images = images.to(config.device)
            angles = angles.to(config.device)

            # Forward pass
            logits = model(images, angles)
            probs = torch.sigmoid(logits)

            predictions.extend(probs.cpu().numpy().flatten())
            ids_list.extend(ids)

    # 6. Create Submission File
    print("Creating Submission File...")
    submission_df = pd.DataFrame({"id": ids_list, "is_iceberg": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)

    # Save
    submission_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")

    # Verify Submission
    print("\nVerifying Submission Format...")
    saved_df = pd.read_csv(config.submission_path)
    print(saved_df.head())

    assert list(saved_df.columns) == [
        "id",
        "is_iceberg",
    ], "Incorrect columns in submission"
    assert len(saved_df) > 0, "Submission file is empty"
    assert (
        saved_df["is_iceberg"].min() >= 0 and saved_df["is_iceberg"].max() <= 1
    ), "Probabilities out of range"

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
