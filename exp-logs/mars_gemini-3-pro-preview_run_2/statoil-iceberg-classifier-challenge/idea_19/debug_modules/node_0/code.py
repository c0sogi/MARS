import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_processed_data, get_fold_loaders
from library.model import DWB_DPN
from library.train import train_one_epoch, validate

if __name__ == "__main__":
    print("Starting Demo Script...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and isolation
    Config.SEED = 42
    Config.EPOCHS = 2  # Reduce epochs for speed
    Config.N_FOLDS = 2  # Reduce folds
    Config.BATCH_SIZE = 16  # Small batch size
    Config.WORKING_DIR = "./working/demo_execution"

    # Update dependent paths
    # Note: CACHE_PATH is a class attribute initialized at import, so we must update it manually
    Config.CACHE_PATH = os.path.join(Config.WORKING_DIR, "cache", "processed_data.npz")

    # Create directories
    os.makedirs(os.path.dirname(Config.CACHE_PATH), exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n[2] Loading and Processing Data...")

    # Load data (force processing to populate our specific demo cache)
    X_train, y_train, inc_train, X_test, inc_test, test_ids = get_processed_data(
        load_cached_data=False
    )

    # Assertions to verify data integrity
    print("    Verifying data shapes...")
    assert (
        len(X_train) == len(y_train) == len(inc_train)
    ), "Training data lengths mismatch"
    assert X_train.shape[1:] == (
        3,
        75,
        75,
    ), f"Unexpected image shape: {X_train.shape[1:]}"
    assert not np.isnan(inc_train).any(), "Incidence angles contain NaNs"

    print(f"    Train samples: {len(X_train)}")
    print(f"    Test samples: {len(X_test)}")

    # ==========================================
    # 3. Model Instantiation & Verification
    # ==========================================
    print("\n[3] Initializing Model...")

    model = DWB_DPN().to(device)

    # Verify Forward Pass
    print("    Verifying forward pass...")
    dummy_img = torch.randn(4, 3, 75, 75).to(device)
    dummy_inc = torch.randn(4).to(device)

    with torch.no_grad():
        output = model(dummy_img, dummy_inc)

    assert output.shape == (4, 1), f"Expected output shape (4, 1), got {output.shape}"
    print("    Forward pass successful. Output shape verified.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n[4] Running Training Loop Demo (Fold 0)...")

    # Get DataLoaders for the first fold
    train_loader, val_loader = get_fold_loaders(
        X_train,
        y_train,
        inc_train,
        fold_idx=0,
        n_folds=Config.N_FOLDS,
        batch_size=Config.BATCH_SIZE,
    )

    # Setup Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Run a short training loop
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_preds = validate(model, val_loader, criterion, device)

        print(
            f"    Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Verify predictions
        assert len(val_preds) == len(
            val_loader.dataset
        ), "Number of predictions does not match validation set size"
        assert all(
            0.0 <= p <= 1.0 for p in val_preds
        ), "Predictions are not valid probabilities"

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n[5] Generating Demo Submission...")

    # Simulate inference on a subset of test data
    model.eval()
    test_subset_ids = test_ids[:10]
    test_subset_preds = []

    # Create a dummy loader for the subset (manually for demo purposes)
    # In production, use get_test_loader
    subset_imgs = torch.FloatTensor(X_test[:10]).to(device)
    subset_incs = torch.FloatTensor(inc_test[:10]).to(device)

    with torch.no_grad():
        out = model(subset_imgs, subset_incs)
        preds = torch.sigmoid(out).cpu().numpy().flatten()
        test_subset_preds = preds

    # Create submission DataFrame
    submission = pd.DataFrame({"id": test_subset_ids, "is_iceberg": test_subset_preds})

    print("    Sample Submission Head:")
    print(submission.head())

    # Verify submission format
    assert list(submission.columns) == [
        "id",
        "is_iceberg",
    ], "Submission columns mismatch"
    assert len(submission) == 10, "Submission length mismatch"

    # Save to demo location
    save_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"    Demo submission saved to {save_path}")

    print("\nDemo completed successfully.")
