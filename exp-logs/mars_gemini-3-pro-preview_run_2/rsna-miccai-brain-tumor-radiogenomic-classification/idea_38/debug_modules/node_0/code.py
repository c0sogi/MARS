import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import process_dataset, BraTSDataset, load_subject_volume
from library.model import AsymmetricEfficientNet
from library.train import train_epoch, validate, predict_tta


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Setting up Configuration for Demo...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.PATIENCE = 1

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"    Device: {device}")
    print(f"    Working Dir: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Processing Verification
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading Pipeline...")

    # Load original metadata
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata not found at {train_meta_path}")

    df_train = pd.read_csv(train_meta_path)

    # Create a small subset for the demo (top 8 samples)
    df_demo_train = df_train.head(8).copy()
    demo_meta_path = os.path.join(Config.WORKING_DIR, "train_demo_subset.csv")
    df_demo_train.to_csv(demo_meta_path, index=False)
    print(f"    Created demo metadata subset with {len(df_demo_train)} samples.")

    # Test single subject volume loading logic
    subject_id = df_demo_train.iloc[0]["BraTS21ID"]
    print(f"    Testing load_subject_volume for BraTS21ID: {subject_id}...")
    volume = load_subject_volume(subject_id, df_demo_train)

    # Assertions for volume shape
    # Expected: (Channels=12, Height=224, Width=224)
    expected_shape = (Config.CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        volume.shape == expected_shape
    ), f"Volume shape mismatch. Expected {expected_shape}, got {volume.shape}"
    assert (
        volume.dtype == np.float32
    ), f"Volume dtype mismatch. Expected float32, got {volume.dtype}"
    print("    Volume loading verification passed.")

    # Process dataset (this handles caching)
    print("    Processing dataset (generating/loading cache)...")
    data, labels = process_dataset(
        demo_meta_path,
        "demo_train",
        load_cached_data=False,  # Force re-compute for demo purposes
    )

    assert len(data) == 8, "Dataset size mismatch."
    assert len(labels) == 8, "Labels size mismatch."
    assert not np.isnan(data).any(), "Data contains NaNs."
    print("    Dataset processing verification passed.")

    # Instantiate PyTorch Dataset
    dataset = BraTSDataset(data, labels, transform=True)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Check batch structure
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (
        Config.BATCH_SIZE,
        Config.CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Batch input shape incorrect."
    assert batch_y.shape == (Config.BATCH_SIZE,), "Batch label shape incorrect."
    print("    DataLoader verification passed.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = AsymmetricEfficientNet().to(device)

    # Run a forward pass with the dummy batch
    with torch.no_grad():
        output = model(batch_x.to(device))

    # Expected output: (Batch_Size, 1) logits
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {output.shape}"
    print("    Model forward pass verification passed.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Verification
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Training Step...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Run one training epoch
    train_loss, train_auc = train_epoch(model, loader, criterion, optimizer, device)

    print(f"    Epoch 1 Result - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN."
    assert 0.0 <= train_auc <= 1.0, "Training AUC out of bounds."

    # Run validation (using same loader for demo simplicity)
    val_loss, val_auc = validate(model, loader, criterion, device)
    print(f"    Validation Result - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # Save demo model
    model_path = os.path.join(Config.WORKING_DIR, "best_model_demo.pth")
    torch.save(model.state_dict(), model_path)
    print("    Training loop verification passed.")

    # --------------------------------------------------------------------------
    # 5. Inference & TTA Verification
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Inference and TTA...")

    # Load test metadata (subset)
    test_meta_path = os.path.join(Config.METADATA_DIR, "test.csv")
    df_test = pd.read_csv(test_meta_path)
    df_demo_test = df_test.head(4).copy()
    demo_test_meta_path = os.path.join(Config.WORKING_DIR, "test_demo_subset.csv")
    df_demo_test.to_csv(demo_test_meta_path, index=False)

    # Process test data
    test_data, _ = process_dataset(
        demo_test_meta_path, "demo_test", load_cached_data=False
    )

    # Create test loader
    # Note: Labels are dummy zeros for test set
    test_labels = np.zeros(len(test_data), dtype=np.float32)
    test_dataset = BraTSDataset(test_data, test_labels, transform=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Run Inference
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            # Verify predict_tta function
            preds = predict_tta(model, inputs, device)

            # Check shape and value range
            assert preds.shape == (inputs.size(0), 1), "TTA output shape mismatch."
            assert (preds >= 0).all() and (
                preds <= 1
            ).all(), "Predictions out of probability range [0, 1]."

            predictions.extend(preds.cpu().numpy().flatten())

    assert len(predictions) == len(
        df_demo_test
    ), "Number of predictions matches test subset size."
    print("    Inference verification passed.")

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    print("\n[6] Generating Demo Submission...")

    submission = pd.DataFrame(
        {"BraTS21ID": df_demo_test["BraTS21ID"], "MGMT_value": predictions}
    )

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    # Verify file existence
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    print("    Submission generation passed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
