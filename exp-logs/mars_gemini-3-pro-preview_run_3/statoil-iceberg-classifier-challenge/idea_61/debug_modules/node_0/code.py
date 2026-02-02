import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_device, print_metric
from library.dataset import get_dataloaders
from library.model import EA_IDPH_CNN
from library.engine import train_one_epoch, evaluate

if __name__ == "__main__":
    # 1. Setup and Configuration
    # Override Config for a quick demonstration run
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32
    # Ensure we use the specific working directory for this idea
    Config.setup_directories()

    # Set reproducible seed
    set_seed(Config.SEED)

    # Get compute device
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    # This will use cached .npy files if available in working/cache, or process json files
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Set to 0 for simple main-thread execution in demo
        load_cached_data=True,
    )

    # Validation: Check batch shapes
    sample_imgs, sample_angles, sample_labels = next(iter(train_loader))
    print(f"Batch Image Shape: {sample_imgs.shape}")  # Should be (B, 3, 75, 75)
    print(f"Batch Angle Shape: {sample_angles.shape}")  # Should be (B, 1)

    assert sample_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Incorrect image batch shape"
    assert sample_angles.shape == (Config.BATCH_SIZE, 1), "Incorrect angle batch shape"
    assert sample_imgs.dtype == torch.float32, "Images should be float32"

    # 3. Model Initialization
    print("Initializing Model (EA_IDPH_CNN)...")
    model = EA_IDPH_CNN().to(device)

    # Validation: Check forward pass logic
    with torch.no_grad():
        dummy_out = model(sample_imgs.to(device), sample_angles.to(device))
        assert dummy_out.shape == (
            Config.BATCH_SIZE,
        ), f"Expected output shape ({Config.BATCH_SIZE},), got {dummy_out.shape}"

    # 4. Training Loop
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print_metric("Train Loss", train_loss)
        print_metric("Val Loss", val_loss)
        print_metric("Val Acc", val_acc)

        # Basic assertion to ensure loss is not NaN
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"

    # 5. Inference on Test Set
    print("Generating predictions on Test set...")
    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for images, angles, img_ids in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            # Forward pass
            outputs = model(images, angles)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            predictions.extend(probs.cpu().numpy())
            ids.extend(img_ids)

    # 6. Create Submission File
    print("Saving submission file...")
    submission_df = pd.DataFrame({"id": ids, "is_iceberg": predictions})

    # Ensure output directory exists (Config.setup_directories() handled this, but being safe)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")

    # 7. Final Validation of Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    loaded_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(loaded_sub) == len(test_loader.dataset), "Submission row count mismatch"
    assert (
        "id" in loaded_sub.columns and "is_iceberg" in loaded_sub.columns
    ), "Missing columns in submission"
    assert (
        loaded_sub["is_iceberg"].min() >= 0 and loaded_sub["is_iceberg"].max() <= 1
    ), "Probabilities out of bounds"

    print("Demo completed successfully.")
