import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.swa_utils import AveragedModel

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data import (
    process_and_cache_data,
    IcebergDataset,
    get_transforms,
)
from library.model import IcebergResNet18
from library.train import train_one_epoch, validate_tta, custom_update_bn


def run_demo():
    print("=== Starting Library Demonstration ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n1. Configuring Environment...")

    # Override Config for the demo to ensure speed and isolation
    Config.WORKING_DIR = "./working/demo_run"
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1
    Config.DEBUG = True
    Config.PRETRAINED = False  # Skip downloading weights for speed

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = device
    print(f"   Device: {device}")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Create working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Data Processing
    # --------------------------------------------------------------------------
    print("\n2. Processing Data...")

    # process_and_cache_data loads JSONs, scales images, handles angles, and saves .npy files.
    # We set load_cached_data=False to demonstrate the processing logic.
    data = process_and_cache_data(load_cached_data=False)

    # Verify data dictionary structure and shapes
    required_keys = [
        "train_images",
        "train_angles",
        "train_labels",
        "test_images",
        "test_angles",
        "test_ids",
    ]
    for key in required_keys:
        assert key in data, f"Missing key in data: {key}"

    # Check image shape: (N, 75, 75, 3) - 3 channels created by adding mean channel
    assert data["train_images"].shape[1:] == (
        75,
        75,
        3,
    ), f"Incorrect train image shape: {data['train_images'].shape}"
    assert data["train_angles"].ndim == 1, "Train angles should be 1D array"

    print("   Data processed and verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Dataset & DataLoader
    # --------------------------------------------------------------------------
    print("\n3. Setting up Datasets and Loaders (Subset)...")

    # Create a small subset for quick demonstration
    subset_size = 32
    X_train_sub = data["train_images"][:subset_size]
    ang_train_sub = data["train_angles"][:subset_size]
    y_train_sub = data["train_labels"][:subset_size]

    X_val_sub = data["train_images"][subset_size : subset_size * 2]
    ang_val_sub = data["train_angles"][subset_size : subset_size * 2]
    y_val_sub = data["train_labels"][subset_size : subset_size * 2]

    # Instantiate Datasets
    train_dataset = IcebergDataset(
        X_train_sub, ang_train_sub, y_train_sub, transform=get_transforms("train")
    )
    val_dataset = IcebergDataset(
        X_val_sub, ang_val_sub, y_val_sub, transform=get_transforms("valid")
    )

    # Instantiate Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # Required for BN stability in some cases
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Verify Batch Shapes
    # Transforms resize images to 224x224
    images, angles, labels = next(iter(train_loader))
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Unexpected image batch shape: {images.shape}"
    assert angles.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected angle batch shape: {angles.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected label batch shape: {labels.shape}"

    print("   Datasets created and batch shapes verified.")

    # --------------------------------------------------------------------------
    # 4. Model Instantiation
    # --------------------------------------------------------------------------
    print("\n4. Initializing Model...")

    model = IcebergResNet18(pretrained=False, dropout_rate=0.5)
    model = model.to(device)

    # Verify Forward Pass
    img_batch = images.to(device)
    ang_batch = angles.to(device)

    with torch.no_grad():
        logits = model(img_batch, ang_batch)

    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Output shape mismatch: {logits.shape}"
    print("   Model initialized and forward pass successful.")

    # --------------------------------------------------------------------------
    # 5. Training Loop
    # --------------------------------------------------------------------------
    print("\n5. Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train for one epoch
    # This uses mixup_data and mixup_criterion internally if Config.USE_MIXUP is True
    avg_loss = train_one_epoch(train_loader, model, optimizer, device, epoch=0)

    print(f"   Training Epoch Completed. Average Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # --------------------------------------------------------------------------
    # 6. Validation (TTA)
    # --------------------------------------------------------------------------
    print("\n6. Running Validation with TTA...")

    # validate_tta performs original + horizontal flip inference and returns log loss
    val_metric = validate_tta(val_loader, model, device)

    print(f"   Validation Completed. Log Loss: {val_metric:.4f}")
    assert not np.isnan(val_metric), "Validation metric is NaN"

    # --------------------------------------------------------------------------
    # 7. SWA Demonstration
    # --------------------------------------------------------------------------
    print("\n7. Demonstrating SWA Components...")

    # Create Averaged Model
    swa_model = AveragedModel(model)

    # Update parameters (Simulating an SWA step)
    swa_model.update_parameters(model)

    # Update Batch Norm statistics
    # This runs a pass over the train loader to update running_mean/var
    print("   Updating SWA Batch Normalization statistics...")
    custom_update_bn(train_loader, swa_model, device)

    # Validate SWA model
    swa_metric = validate_tta(val_loader, swa_model, device)
    print(f"   SWA Validation Log Loss: {swa_metric:.4f}")

    # --------------------------------------------------------------------------
    # 8. Prediction
    # --------------------------------------------------------------------------
    print("\n8. Generating Predictions (Test Subset)...")

    # Prepare test subset
    X_test_sub = data["test_images"][:subset_size]
    ang_test_sub = data["test_angles"][:subset_size]
    ids_test_sub = data["test_ids"][:subset_size]

    test_dataset = IcebergDataset(
        X_test_sub, ang_test_sub, transform=get_transforms("test")
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model.eval()
    preds = []

    with torch.no_grad():
        for imgs, angs in test_loader:
            imgs = imgs.to(device)
            angs = angs.to(device)

            # Simple inference (without TTA for this specific block, though predict_test uses TTA)
            out = model(imgs, angs)
            probs = torch.sigmoid(out)
            preds.extend(probs.cpu().numpy().flatten())

    preds = np.array(preds)

    assert len(preds) == len(ids_test_sub), "Prediction count mismatch"
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions out of probability range"

    print(f"   Generated {len(preds)} predictions.")
    print(f"   Sample predictions: {preds[:5]}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
