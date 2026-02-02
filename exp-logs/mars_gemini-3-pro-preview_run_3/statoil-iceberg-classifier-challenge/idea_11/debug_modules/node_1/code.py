import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.data import prepare_data, IcebergDataset, get_transforms
from library.model import SHMP_CNN
from library.train import train_one_epoch, validate


def demo_pipeline():
    print("=== Starting Library Usage Demo ===")

    # 1. Reproducibility
    # Set seed using the library utility
    set_seed(Config.SEED)
    print(f"Seed set to {Config.SEED}")

    # 2. Data Loading
    # We use the prepare_data function which handles caching.
    # Since the environment has cached files in ./working/idea_11, this should be fast.
    print("\n--- Data Preparation ---")
    try:
        (train_data, val_data, test_data) = prepare_data(load_cached_data=True)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Unpack training data
    X_train, angle_train, y_train, ids_train = train_data

    # 3. Create a Mini-Batch Subset for Speed
    # We select just 32 samples to demonstrate functionality without long wait times.
    subset_size = 32
    print(f"Creating a subset of {subset_size} samples for demonstration...")

    X_mini = X_train[:subset_size]
    angle_mini = angle_train[:subset_size]
    y_mini = y_train[:subset_size]
    ids_mini = ids_train[:subset_size]

    # 4. Dataset and DataLoader
    print("\n--- Dataset & DataLoader ---")
    # Instantiate the custom Dataset class
    dataset = IcebergDataset(
        X=X_mini,
        angles=angle_mini,
        y=y_mini,
        ids=ids_mini,
        transform=get_transforms("train"),
    )

    # Verify dataset length
    assert (
        len(dataset) == subset_size
    ), f"Dataset length {len(dataset)} != {subset_size}"

    # Create DataLoader
    batch_size = 8
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for simple demo to avoid multiprocessing overhead
    )

    # Fetch one batch to verify shapes
    images, angles, labels = next(iter(loader))
    print(
        f"Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions to ensure data pipeline is correct
    # Image: (Batch, 3 Channels, 75 Height, 75 Width)
    assert images.shape == (batch_size, 3, 75, 75), "Incorrect Image Batch Shape"
    # Angle: (Batch,)
    assert angles.shape == (batch_size,), "Incorrect Angle Batch Shape"
    # Label: (Batch,)
    assert labels.shape == (batch_size,), "Incorrect Label Batch Shape"

    # 5. Model Initialization
    print("\n--- Model Initialization ---")
    device = Config.DEVICE
    print(f"Using device: {device}")

    model = SHMP_CNN().to(device)

    # 6. Forward Pass Verification
    print("Verifying Forward Pass...")
    images = images.to(device)
    angles = angles.to(device)

    # Run inference (no grad)
    with torch.no_grad():
        outputs = model(images, angles)

    print(f"Model Output Shape: {outputs.shape}")
    # Output should be (Batch, 1) logits
    assert outputs.shape == (
        batch_size,
        1,
    ), "Model output shape mismatch. Expected (B, 1)."

    # 7. Training Loop Demo
    print("\n--- Training Step Demo ---")
    # Setup standard components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run one epoch of training using the library function
    # This updates model weights
    train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
    print(f"Train Loss (1 Epoch): {train_loss:.6f}")

    # Verify loss is a valid number
    assert isinstance(train_loss, float), "Train loss must be a float"
    assert not np.isnan(train_loss), "Train loss returned NaN"

    # 8. Validation Loop Demo
    print("\n--- Validation Step Demo ---")
    # Run validation using the library function
    val_loss = validate(model, loader, criterion, device)
    print(f"Validation Loss: {val_loss:.6f}")

    assert isinstance(val_loss, float), "Validation loss must be a float"

    # 9. Checkpoint Management Demo
    print("\n--- Checkpoint Management ---")
    # Use a temporary directory for this demo
    demo_ckpt_dir = os.path.join(Config.WORKING_DIR, "demo_checkpoints")

    # Fake state dictionary
    state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": val_loss,
        "fold": 999,
    }

    # Save checkpoint
    print(f"Saving checkpoint to {demo_ckpt_dir}...")
    save_checkpoint(state, is_best=True, fold_idx=999, checkpoint_dir=demo_ckpt_dir)

    expected_path = os.path.join(demo_ckpt_dir, "checkpoint_fold_999.pth")
    expected_best_path = os.path.join(demo_ckpt_dir, "model_best_fold_999.pth")

    assert os.path.exists(expected_path), "Checkpoint file was not created."
    assert os.path.exists(expected_best_path), "Best model file was not created."

    # Load checkpoint
    print("Loading checkpoint back...")
    model_new = SHMP_CNN().to(device)
    loaded_state = load_checkpoint(expected_path, model_new)

    # Verify loaded metadata
    assert loaded_state["fold"] == 999, "Loaded checkpoint metadata mismatch (fold)."
    assert (
        loaded_state["best_loss"] == val_loss
    ), "Loaded checkpoint metadata mismatch (loss)."
    print("Checkpoint save/load verified.")

    # Cleanup
    if os.path.exists(demo_ckpt_dir):
        shutil.rmtree(demo_ckpt_dir)
        print("Temporary checkpoint directory cleaned up.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    demo_pipeline()
