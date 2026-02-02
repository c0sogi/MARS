import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import provided library modules
from library import config, utils, dataset, model, engine


def run_demo_pipeline():
    print("=== Starting Demonstration Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override configuration for a fast demonstration
    config.NUM_EPOCHS = 2
    config.BATCH_SIZE = 32
    config.NUM_FOLDS = 3
    config.WORKING_DIR = "./working/demo_run"

    # Update derived paths in config manually since they were initialized at import
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    config.CHECKPOINT_DIR = os.path.join(config.WORKING_DIR, "checkpoints")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Update cache paths to point to the demo directory
    config.CACHE_PATH_X_TRAIN = os.path.join(config.WORKING_DIR, "X_train.npy")
    config.CACHE_PATH_Y_TRAIN = os.path.join(config.WORKING_DIR, "y_train.npy")
    config.CACHE_PATH_ANGLE_TRAIN = os.path.join(config.WORKING_DIR, "angles_train.npy")

    config.CACHE_PATH_X_TEST = os.path.join(config.WORKING_DIR, "X_test.npy")
    config.CACHE_PATH_IDS_TEST = os.path.join(config.WORKING_DIR, "ids_test.npy")
    config.CACHE_PATH_ANGLE_TEST = os.path.join(config.WORKING_DIR, "angles_test.npy")

    # Set seed for reproducibility
    utils.seed_everything(config.SEED)
    print(f"Configuration updated. Working directory: {config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n--- Loading Data ---")
    # We use fold 0. load_cached_data=False ensures we process raw data for this demo
    train_loader, val_loader = dataset.get_data_loaders(
        fold_idx=0, load_cached_data=False
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")

    # Verify Train Batch
    images, angles, labels = next(iter(train_loader))
    print(
        f"Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions to ensure data pipeline is correct
    assert images.shape == (config.BATCH_SIZE, 3, 75, 75), "Incorrect image batch shape"
    assert angles.shape == (config.BATCH_SIZE,), "Incorrect angle batch shape"
    assert labels.shape == (config.BATCH_SIZE,), "Incorrect label batch shape"
    assert not torch.isnan(images).any(), "NaNs found in images"
    # Angles might have been imputed, check for NaNs just in case
    assert not torch.isnan(angles).any(), "NaNs found in angles after imputation"

    # Verify Cache Creation
    assert os.path.exists(config.CACHE_PATH_X_TRAIN), "X_train cache not created"
    print("Data loaded and verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Verification
    # -------------------------------------------------------------------------
    print("\n--- Initializing Model ---")
    net = model.SHH_SE_CNN().to(config.DEVICE)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_out = net(images.to(config.DEVICE), angles.to(config.DEVICE))

    print(f"Model Output Shape: {dummy_out.shape}")
    assert dummy_out.shape == (config.BATCH_SIZE,), "Model output shape mismatch"
    print("Model initialized and verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\n--- Starting Training Loop ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    es = engine.EarlyStopping(patience=2, fold_idx=0)

    for epoch in range(config.NUM_EPOCHS):
        train_loss = engine.train_one_epoch(
            net, train_loader, criterion, optimizer, config.DEVICE
        )
        val_loss = engine.evaluate(net, val_loader, criterion, config.DEVICE)

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        es(val_loss, net, optimizer, epoch)
        if es.early_stop:
            print("Early stopping triggered.")
            break

    print("Training complete.")

    # -------------------------------------------------------------------------
    # 5. Checkpoint Loading
    # -------------------------------------------------------------------------
    print("\n--- Loading Best Checkpoint ---")
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "model_best_fold_0.pth")
    # Fallback to last checkpoint if best wasn't saved (e.g. loss didn't improve)
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint_fold_0.pth")

    checkpoint = utils.load_checkpoint(net, best_model_path)
    print(
        f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['best_loss']:.4f}"
    )

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n--- Running Inference ---")
    test_loader = dataset.get_test_loader(load_cached_data=False)

    net.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, angles, ids in test_loader:
            images = images.to(config.DEVICE)
            angles = angles.to(config.DEVICE)

            logits = net(images, angles)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_ids.extend(ids)

    # Create Submission DataFrame
    submission = pd.DataFrame({"id": all_ids, "is_iceberg": all_probs})

    # Save Submission
    sub_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")
    print(f"Submission shape: {submission.shape}")
    print("First 5 rows:")
    print(submission.head())

    # Final Validation
    assert len(submission) == 321, f"Expected 321 predictions, got {len(submission)}"
    assert (
        submission["is_iceberg"].min() >= 0 and submission["is_iceberg"].max() <= 1
    ), "Probabilities out of range"

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo_pipeline()
