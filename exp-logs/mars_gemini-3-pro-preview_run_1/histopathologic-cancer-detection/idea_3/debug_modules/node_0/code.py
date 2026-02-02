import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, compute_auc, get_logger
from library.dataset import TumorDataset, get_transforms
from library.model import ConvNeXtTinyCustom
from library.engine import train_one_epoch, validate, predict_tta


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print("Initializing Demonstration...")

    # Override Config parameters to run a fast "unit test" style execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    # Create working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Device configuration
    device = Config.DEVICE
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Dataset Verification
    # -------------------------------------------------------------------------
    print("\n--- Testing Data Loading ---")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create small subsets for demonstration (16 samples each)
    subset_size = 16
    train_subset = train_df.head(subset_size).copy()
    val_subset = val_df.head(subset_size).copy()
    test_subset = test_df.head(subset_size).copy()

    print(f"Created subsets of size {subset_size} for Train, Val, and Test.")

    # Instantiate Datasets
    train_ds = TumorDataset(train_subset, transforms=get_transforms("train"))
    val_ds = TumorDataset(val_subset, transforms=get_transforms("val"))
    test_ds = TumorDataset(test_subset, transforms=get_transforms("test"))

    # Verify Dataset __getitem__
    sample_img, sample_label = train_ds[0]

    # Check Image Shape: Should be (3, 48, 48) after transforms (CROP_SIZE=48)
    expected_shape = (3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert (
        sample_img.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {sample_img.shape}"

    # Check Label Type
    assert isinstance(sample_label, torch.Tensor), "Label should be a torch.Tensor"
    assert sample_label.dtype == torch.float32, "Label should be float32"

    print("Dataset __getitem__ verification passed.")

    # Instantiate DataLoaders
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Verify DataLoader batch
    batch_imgs, batch_labels = next(iter(train_loader))
    assert batch_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Batch image shape mismatch: {batch_imgs.shape}"
    assert batch_labels.shape == (
        Config.BATCH_SIZE,
    ), f"Batch label shape mismatch: {batch_labels.shape}"

    print("DataLoader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Verification
    # -------------------------------------------------------------------------
    print("\n--- Testing Model Architecture ---")

    # Initialize model
    # Using pretrained=False for speed in demo (avoids downloading weights if not cached)
    model = ConvNeXtTinyCustom(pretrained=False)
    model.to(device)

    # Forward pass verification
    dummy_input = torch.randn(
        Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch_Size, 1)
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {output.shape}"

    print("Model instantiation and forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Engine Verification
    # -------------------------------------------------------------------------
    print("\n--- Testing Training Engine ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Test train_one_epoch
    initial_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

    assert isinstance(initial_loss, float), "train_one_epoch should return a float loss"
    assert not np.isnan(initial_loss), "Training loss returned NaN"
    print(f"Training step successful. Loss: {initial_loss:.4f}")

    # Test validate
    val_loss, val_auc = validate(model, val_loader, criterion, device)

    assert isinstance(val_loss, float), "Validation loss should be a float"
    assert isinstance(val_auc, float), "Validation AUC should be a float"
    assert 0.0 <= val_auc <= 1.0, "AUC score out of range [0, 1]"

    print(f"Validation step successful. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # -------------------------------------------------------------------------
    # 5. Inference Verification (TTA)
    # -------------------------------------------------------------------------
    print("\n--- Testing Inference (TTA) ---")

    # Save the current model state to test loading logic
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved"

    # Predict using TTA
    predictions = predict_tta(model, test_loader, device)

    # Verify predictions
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
    assert len(predictions) == len(
        test_subset
    ), f"Prediction count mismatch. Expected {len(test_subset)}, got {len(predictions)}"
    assert (
        predictions.min() >= 0.0 and predictions.max() <= 1.0
    ), "Predictions should be probabilities between 0 and 1"

    print(f"Inference successful. Generated {len(predictions)} predictions.")

    # -------------------------------------------------------------------------
    # 6. Utility Verification
    # -------------------------------------------------------------------------
    print("\n--- Testing Utilities ---")

    # Test compute_auc manually
    y_true = np.array([0, 0, 1, 1])
    y_pred_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    y_pred_bad = np.array([0.9, 0.8, 0.2, 0.1])

    auc_perfect = compute_auc(y_true, y_pred_perfect)
    auc_bad = compute_auc(y_true, y_pred_bad)

    assert auc_perfect == 1.0, f"Expected AUC 1.0, got {auc_perfect}"
    assert auc_bad == 0.0, f"Expected AUC 0.0, got {auc_bad}"

    print("Utility functions verification passed.")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    run_demo()
