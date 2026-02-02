import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.cuda.amp import GradScaler

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import (
    get_loaders,
    decode_polyline,
    process_annotations,
    CatheterDataset,
    get_transforms,
)
from library.model import CatheterModel
from library.loss import CustomLoss
from library.train import train_one_epoch, valid_one_epoch


def run_demonstration():
    print("Starting Catheter Detection Library Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")
    # Override Config defaults to run a fast, lightweight demo
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Use a tiny subset
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.PRETRAINED = False  # Skip downloading weights for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Setup directories
    Config.setup()

    # Set seeds
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test get_score (AUC calculation)
    # Create dummy data: 10 samples, 11 classes
    y_true = np.random.randint(0, 2, (10, 11))
    # Ensure at least one class has both 0 and 1 to avoid AUC errors
    y_true[:, 0] = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.random.rand(10, 11)

    auc_score = get_score(y_true, y_pred)
    print(f"    Calculated Dummy AUC: {auc_score:.4f}")

    if auc_score == 0.0:
        # It's possible random data results in undefined AUC for all columns if splits are bad,
        # but with the fixed column 0, it should be > 0.
        pass
    assert 0.0 <= auc_score <= 1.0, "AUC score out of range [0, 1]"

    # -------------------------------------------------------------------------
    # 3. Verify Data Pipeline
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Pipeline...")

    # Test Polyline Decode
    # Polyline: [[0, 0], [5, 5]] -> A line from top-left to center
    dummy_polyline = "[[0, 0], [5, 5]]"
    dummy_shape = (10, 10)
    mask = decode_polyline(dummy_polyline, dummy_shape, thickness=1)

    assert (
        mask.shape == dummy_shape
    ), f"Mask shape mismatch. Expected {dummy_shape}, got {mask.shape}"
    # Check that some pixels are drawn (sum > 0)
    assert np.sum(mask) > 0, "Mask sum mismatch. Expected > 0"
    print("    decode_polyline logic verified.")

    # Test Data Loaders
    print("    Initializing DataLoaders (Debug Mode)...")
    train_loader, val_loader = get_loaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force processing
        debug=Config.DEBUG,
    )

    # Fetch one batch
    images, labels, masks = next(iter(train_loader))

    print(
        f"    Batch Shapes -> Images: {images.shape}, Labels: {labels.shape}, Masks: {masks.shape}"
    )

    # Assertions
    # Images: (B, 3, H, W)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    )
    # Labels: (B, NumClasses)
    assert labels.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)
    # Masks: (B, 1, H, W)
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    )

    # -------------------------------------------------------------------------
    # 4. Verify Model and Loss
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model and Loss...")

    model = CatheterModel(pretrained=Config.PRETRAINED)
    model.to(device)

    # Move batch to device
    images = images.to(device, dtype=torch.float32)
    labels = labels.to(device, dtype=torch.float32)
    masks = masks.to(device, dtype=torch.float32)

    # Forward Pass
    logits, mask_preds = model(images)

    print(
        f"    Model Output Shapes -> Logits: {logits.shape}, Mask Preds: {mask_preds.shape}"
    )

    assert logits.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)
    assert mask_preds.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    )

    # Loss Calculation
    criterion = CustomLoss()
    loss = criterion(logits, mask_preds, labels, masks)

    print(f"    Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # Verify Backward Pass (Gradient Check)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()

    # Check if gradients are populated for a key layer (e.g., final fc)
    assert model.fc.weight.grad is not None, "Gradients not computed for FC layer"
    print("    Backward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Demonstrate Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Training Loop (1 Epoch)...")

    scaler = GradScaler()

    # Train One Epoch
    print("    Running train_one_epoch...")
    train_loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloader=train_loader,
        device=device,
        epoch=0,
        scaler=scaler,
    )
    print(f"    -> Train Loss: {train_loss:.6f}")

    # Valid One Epoch
    print("    Running valid_one_epoch...")
    val_loss, val_auc = valid_one_epoch(
        model=model, criterion=criterion, dataloader=val_loader, device=device
    )
    print(f"    -> Val Loss: {val_loss:.6f}")
    print(f"    -> Val AUC: {val_auc}")

    print("\nDemonstration complete successfully.")


if __name__ == "__main__":
    run_demonstration()
