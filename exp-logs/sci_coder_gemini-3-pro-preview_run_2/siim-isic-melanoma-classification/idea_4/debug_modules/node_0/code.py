import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import ISICDataset, get_transforms, process_metadata
from library.model import HybridEfficientNet
from library.engine import train_one_epoch, evaluate, predict


def main():
    # 1. Setup and Configuration Override
    # We override specific Config attributes to ensure a fast demonstration run.
    print("Setting up configuration for demo run...")
    seed_everything(Config.SEED)

    # Enable debug mode to use a small subset (500 samples)
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8  # Small batch size for demo
    Config.NUM_WORKERS = 2  # Reduce workers for lighter execution

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Processing
    print("\n[Step 1] Processing Metadata...")
    # Force processing from scratch to verify logic, debug=True triggers subsampling
    train_data, val_data, test_data = process_metadata(
        load_cached_data=False, debug=True
    )

    # Verification: Check data structures
    assert (
        len(train_data["image_paths"]) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} train samples, got {len(train_data['image_paths'])}"
    assert train_data["meta_features"].shape[0] == Config.DEBUG_SAMPLE_SIZE
    assert train_data["targets"].shape[0] == Config.DEBUG_SAMPLE_SIZE

    meta_dim = train_data["meta_features"].shape[1]
    print(f"Metadata processed successfully. Input feature dimension: {meta_dim}")

    # 3. Dataset and DataLoader Instantiation
    print("\n[Step 2] Creating Datasets and Loaders...")

    train_dataset = ISICDataset(
        image_paths=train_data["image_paths"],
        meta_features=train_data["meta_features"],
        targets=train_data["targets"],
        aux_targets=train_data["aux_targets"],
        transform=get_transforms(data="train"),
    )

    val_dataset = ISICDataset(
        image_paths=val_data["image_paths"],
        meta_features=val_data["meta_features"],
        targets=val_data["targets"],
        aux_targets=val_data["aux_targets"],
        transform=get_transforms(data="valid"),
    )

    test_dataset = ISICDataset(
        image_paths=test_data["image_paths"],
        meta_features=test_data["meta_features"],
        targets=None,
        aux_targets=None,
        transform=get_transforms(data="test"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Verification: Fetch one batch to check shapes
    sample_batch = next(iter(train_loader))
    images = sample_batch["image"]
    meta = sample_batch["meta"]
    targets = sample_batch["target"]

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image tensor shape: {images.shape}"
    assert meta.shape == (
        Config.BATCH_SIZE,
        meta_dim,
    ), f"Incorrect meta tensor shape: {meta.shape}"
    print("DataLoaders created and verified successfully.")

    # 4. Model Initialization
    print("\n[Step 3] Initializing Hybrid Model...")
    model = HybridEfficientNet(meta_dim=meta_dim, pretrained=True)
    model.to(device)

    # Verification: Forward pass
    with torch.no_grad():
        logits_mal, logits_diag = model(images.to(device), meta.to(device))

    assert logits_mal.shape == (
        Config.BATCH_SIZE,
        1,
    ), "Malignancy logits shape mismatch"
    assert logits_diag.shape == (
        Config.BATCH_SIZE,
        Config.NUM_AUX_CLASSES,
    ), "Diagnosis logits shape mismatch"
    print("Model initialized and forward pass verified.")

    # 5. Training Loop
    print("\n[Step 4] Running Training (1 Epoch)...")
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch=1)

    # Verification: Check metrics
    assert not np.isnan(train_metrics["loss_total"]), "Training loss is NaN"
    print(f"Training complete. Loss: {train_metrics['loss_total']:.4f}")

    # 6. Evaluation
    print("\n[Step 5] Running Evaluation...")
    val_loss, val_auc = evaluate(model, val_loader, device)

    # Verification: Check AUC range
    assert 0.0 <= val_auc <= 1.0, f"AUC score out of range: {val_auc}"
    print(f"Evaluation complete. Val AUC: {val_auc:.4f}")

    # 7. Prediction
    print("\n[Step 6] Generating Test Predictions...")
    test_preds = predict(model, test_loader, device)

    # Verification: Check predictions shape and range
    assert len(test_preds) == len(
        test_dataset
    ), f"Prediction count mismatch. Expected {len(test_dataset)}, got {len(test_preds)}"
    assert (test_preds >= 0).all() and (
        test_preds <= 1
    ).all(), "Predictions contain values outside [0, 1]"
    print(f"Predictions generated for {len(test_preds)} test samples.")

    # 8. Submission File Generation
    print("\n[Step 7] Creating Submission File...")
    # Load test metadata to get image names
    df_test = pd.read_csv(Config.TEST_META)

    # In debug mode, we only have predictions for the subset
    if Config.DEBUG:
        df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE]

    submission = pd.DataFrame(
        {"image_name": df_test["image_name"], "target": test_preds}
    )

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verification: Check file existence
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
