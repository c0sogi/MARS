import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import (
    DEVICE,
    NUM_WORKERS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MASTER_SEED,
    WORKING_DIR,
    SUBMISSION_DIR,
)
from library.utils import seed_everything
from library.dataset import get_datasets
from library.model import NarrowSEResNet
from library.engine import train_model, predict_tta


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Reproducibility
    seed_everything(MASTER_SEED)
    print(f"Random seed set to {MASTER_SEED}")

    # 2. Data Loading
    print("Loading datasets...")
    # We use load_cached_data=False to demonstrate loading from metadata/disk directly
    # In a real repeated run, True would be faster.
    train_ds, val_ds, test_ds, test_ids = get_datasets(load_cached_data=False)

    print(f"Train set size: {len(train_ds)}")
    print(f"Val set size:   {len(val_ds)}")
    print(f"Test set size:  {len(test_ds)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Verification: Check batch shapes
    dummy_images, dummy_labels = next(iter(train_loader))
    print(f"Batch Image Shape: {dummy_images.shape}")
    print(f"Batch Label Shape: {dummy_labels.shape}")

    # Assertions for data integrity
    assert dummy_images.shape[1:] == (3, 32, 32), "Incorrect image dimensions"
    assert dummy_labels.ndim == 1, "Labels should be 1D tensor"

    # 3. Model Initialization
    print("Initializing model...")
    model = NarrowSEResNet().to(DEVICE)

    # Verification: Forward pass with dummy data
    dummy_input = torch.randn(2, 3, 32, 32).to(DEVICE)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, 1), "Model output shape mismatch (expected (B, 1))"

    # 4. Training Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Simple scheduler for demo
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

    # Run Training (Reduced epochs for demonstration speed)
    print("Starting training loop (2 epochs for demo)...")
    model_save_path = "model_demo.pth"

    best_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_epochs=2,  # Reduced from config.EPOCHS for speed
        patience=2,  # Reduced patience
        min_delta=1e-4,
        model_filename=model_save_path,
    )
    print(f"Training complete. Best Validation Loss: {best_loss:.4f}")

    # 5. Inference
    print("Running inference on test set with TTA...")
    # Load best model weights (though current model is likely best in this short run)
    # We use the one saved by train_model
    checkpoint_path = os.path.join(WORKING_DIR, model_save_path)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    preds = predict_tta(model, test_loader, DEVICE)

    print(f"Predictions shape: {preds.shape}")

    # Verification: Check predictions range
    assert preds.shape[0] == len(
        test_ids
    ), "Number of predictions does not match number of test IDs"
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    # 6. Submission Generation
    print("Generating submission file...")
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": preds.flatten()})

    # Ensure output directory exists (handled by config, but good practice)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Preview
    print("Submission Head:")
    print(submission_df.head())

    print("=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
