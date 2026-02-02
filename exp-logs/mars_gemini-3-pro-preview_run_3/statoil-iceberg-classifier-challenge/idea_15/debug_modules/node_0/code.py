import os
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

# Import from the provided library files
from library.dataset import load_data, IcebergDataset, get_transforms, set_seed
from library.model import SHSE_CNN
from library.trainer import train_fold


def run_demo():
    # 1. Configuration and Setup
    print("--- Starting Library Usage Demo ---")
    WORK_DIR = "./working/demo_usage"
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

    # Clean up previous run if exists
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading Demonstration
    print("\n[1/5] Loading Data...")
    # load_data caches processed numpy arrays.
    # We load 'train' data which includes images, angles, and labels.
    X_all, angles_all, y_all, ids_all = load_data(mode="train")

    # Verification: Check shapes
    assert len(X_all) == len(y_all), "Mismatch between images and labels count"
    assert len(X_all) == len(angles_all), "Mismatch between images and angles count"
    assert X_all.shape[1:] == (75, 75, 3), f"Unexpected image shape: {X_all.shape}"
    print(f"Successfully loaded {len(X_all)} samples.")

    # 3. Dataset and DataLoader Demonstration
    print("\n[2/5] Preparing Datasets (Subset for Speed)...")

    # Create a small subset for demonstration purposes (64 samples)
    subset_size = 64
    train_size = 48
    val_size = 16

    X_subset = X_all[:subset_size]
    angles_subset = angles_all[:subset_size]
    y_subset = y_all[:subset_size]

    # Split into train/val for the demo
    X_train = X_subset[:train_size]
    a_train = angles_subset[:train_size]
    y_train = y_subset[:train_size]

    X_val = X_subset[train_size:]
    a_val = angles_subset[train_size:]
    y_val = y_subset[train_size:]

    # Instantiate IcebergDataset with transforms
    # Note: get_transforms("train") applies augmentation, "val" does not.
    train_ds = IcebergDataset(
        X_train, a_train, y_train, transform=get_transforms("train")
    )
    val_ds = IcebergDataset(X_val, a_val, y_val, transform=get_transforms("val"))

    # Create DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches: {len(val_loader)}")

    # 4. Model Instantiation and Forward Pass Verification
    print("\n[3/5] Initializing SHSE_CNN Model...")
    model = SHSE_CNN().to(device)

    # Fetch a single batch to verify forward pass
    dummy_imgs, dummy_angles, dummy_labels = next(iter(train_loader))
    dummy_imgs = dummy_imgs.to(device)
    dummy_angles = dummy_angles.to(device)

    # Run forward pass
    with torch.no_grad():
        output = model(dummy_imgs, dummy_angles)

    # Verification: Output shape should be (Batch_Size, 1)
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch. Expected {(batch_size, 1)}, got {output.shape}"
    print("Model forward pass successful. Output shape verified.")

    # 5. Training Loop Demonstration (using library.trainer)
    print("\n[4/5] Running Training Loop (2 Epochs)...")

    # train_fold handles the training loop, validation, and checkpointing
    # We use a very short training duration for the demo
    best_loss, model_path = train_fold(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        fold_idx=0,
        epochs=2,  # Limit epochs for speed
        patience=2,
        lr=1e-3,
        weight_decay=1e-4,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    print(f"Training complete. Best Val Loss: {best_loss:.4f}")
    print(f"Model saved to: {model_path}")

    # Verification: Check if checkpoint file exists
    assert os.path.exists(model_path), "Checkpoint file was not created."

    # 6. Inference Demonstration
    print("\n[5/5] Verifying Inference with Saved Model...")

    # Load model state
    loaded_model = SHSE_CNN().to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()

    # Predict on validation set
    preds = []
    with torch.no_grad():
        for imgs, angs, _ in val_loader:
            imgs = imgs.to(device)
            angs = angs.to(device)
            logits = loaded_model(imgs, angs)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds)

    # Verification: Check predictions
    assert (
        len(preds) == val_size
    ), f"Prediction count mismatch. Expected {val_size}, got {len(preds)}"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    print("Inference successful. Predictions range verified.")
    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    try:
        run_demo()
    except AssertionError as e:
        print(f"\nDEMO FAILED: Assertion Error - {e}")
        exit(1)
    except Exception as e:
        print(f"\nDEMO FAILED: An unexpected error occurred - {e}")
        exit(1)
