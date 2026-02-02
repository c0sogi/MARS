import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Import library components
from library.config import Config
from library.utils import seed_everything, rle_decode, get_auc_score
from library.dataset import get_dataloader, CatheterDataset
from library.model import MultiTaskEfficientNet
from library.loss import MultiTaskLoss
from library.engine import train_one_epoch, valid_one_epoch, inference_fn


def run_demo():
    print("Starting Library Usage Demonstration...")

    # ==========================================
    # 1. Setup and Configuration Overrides
    # ==========================================
    print("\n[1] Setting up configuration and environment...")
    seed_everything(Config.SEED)

    # Override Config for speed and memory efficiency during demo
    Config.IMAGE_SIZE = 256
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1

    # Ensure working directory exists (as per Config)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"    Device: {Config.DEVICE}")
    print(f"    Image Size: {Config.IMAGE_SIZE}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\n[2] Verifying Utility Functions...")

    # Test rle_decode
    # RLE: "1 3" -> means start at pixel 1 (1-based), length 3.
    # In a 3x3 flattened image (indices 0..8), 1-based index 1 is 0-based index 0.
    # So "1 3" -> pixels 0, 1, 2 should be 1.
    dummy_rle = "1 3"
    dummy_shape = (3, 3)
    decoded_mask = rle_decode(dummy_rle, dummy_shape)

    expected_mask = np.zeros((3, 3), dtype=np.uint8)
    expected_mask[0, 0] = 1
    expected_mask[1, 0] = 1  # Fortran order (column-major) logic in rle_decode
    expected_mask[2, 0] = 1

    # rle_decode implementation uses 'F' order reshape.
    # "1 3" -> indices 0, 1, 2 in flattened array.
    # Reshape (3,3) order='F':
    # Index 0 -> (0,0)
    # Index 1 -> (1,0)
    # Index 2 -> (2,0)

    assert (
        decoded_mask.shape == dummy_shape
    ), f"Expected shape {dummy_shape}, got {decoded_mask.shape}"
    assert np.array_equal(decoded_mask, expected_mask), "RLE decoding logic mismatch."
    print("    rle_decode: Verified.")

    # Test get_auc_score
    y_true = np.array([[0, 1], [1, 0], [0, 1]])
    y_pred = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    # Perfect prediction
    score = get_auc_score(y_true, y_pred)
    assert score == 1.0, f"Expected AUC 1.0, got {score}"
    print("    get_auc_score: Verified.")

    # ==========================================
    # 3. Verify Dataset and DataLoader
    # ==========================================
    print("\n[3] Verifying Dataset and DataLoader...")

    # Create a small subset dataloader
    # We use sample_size=16 to create 4 batches of size 4
    train_loader = get_dataloader(
        metadata_path=Config.TRAIN_METADATA,
        mode="train",
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Deterministic for demo
        num_workers=0,  # Avoid multiprocessing overhead for small demo
        sample_size=16,
    )

    print(f"    DataLoader created with {len(train_loader)} batches.")

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    masks = batch["mask"]
    targets = batch["targets"]
    mask_validity = batch["mask_validity"]
    uids = batch["StudyInstanceUID"]

    # Verify Shapes
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Image shape mismatch: {images.shape}"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Mask shape mismatch: {masks.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Target shape mismatch: {targets.shape}"
    assert mask_validity.shape == (
        Config.BATCH_SIZE,
    ), f"Mask validity shape mismatch: {mask_validity.shape}"

    print("    Batch shapes verified.")
    print("    Data loading pipeline: Verified.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n[4] Verifying Model Architecture...")

    # Initialize model (pretrained=False for speed/offline demo)
    model = MultiTaskEfficientNet(pretrained=False)
    model.to(Config.DEVICE)

    # Forward pass
    images = images.to(Config.DEVICE)
    cls_logits, seg_logits = model(images)

    # Check outputs
    assert cls_logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Cls logits shape mismatch: {cls_logits.shape}"
    assert seg_logits.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Seg logits shape mismatch: {seg_logits.shape}"

    print("    Forward pass successful.")
    print("    Model architecture: Verified.")

    # ==========================================
    # 5. Verify Loss Function
    # ==========================================
    print("\n[5] Verifying Loss Function...")

    criterion = MultiTaskLoss(
        load_cached_data=False
    )  # Force re-compute weights for demo

    targets = targets.to(Config.DEVICE)
    masks = masks.to(Config.DEVICE)
    mask_validity = mask_validity.to(Config.DEVICE)

    loss_dict = criterion(cls_logits, targets, seg_logits, masks, mask_validity)

    assert "loss" in loss_dict, "Total loss missing from loss dict"
    assert "cls_loss" in loss_dict, "Classification loss missing from loss dict"
    assert "seg_loss" in loss_dict, "Segmentation loss missing from loss dict"

    total_loss = loss_dict["loss"]
    assert total_loss.ndim == 0, "Loss should be a scalar"
    assert not torch.isnan(total_loss), "Loss is NaN"

    # Check backward capability
    total_loss.backward()
    print("    Backward pass successful.")
    print("    Loss function: Verified.")

    # ==========================================
    # 6. Verify Training Engine
    # ==========================================
    print("\n[6] Verifying Training Engine...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Run one epoch of training
    # Note: We use the same small loader created earlier
    epoch_loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        device=Config.DEVICE,
        epoch=0,
    )

    print(f"    Train Epoch Loss: {epoch_loss:.4f}")
    assert epoch_loss > 0, "Training loss should be positive"

    # Run one epoch of validation
    # Create val loader (reuse train metadata for demo simplicity, but with val mode)
    val_loader = get_dataloader(
        metadata_path=Config.TRAIN_METADATA,
        mode="val",
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        sample_size=16,
    )

    val_loss, val_auc = valid_one_epoch(model, val_loader, Config.DEVICE)
    print(f"    Val Epoch Loss: {val_loss:.4f}")
    print(f"    Val AUC: {val_auc:.4f}")

    assert val_loss > 0, "Validation loss should be positive"
    assert 0 <= val_auc <= 1, "AUC should be between 0 and 1"

    print("    Training engine: Verified.")

    # ==========================================
    # 7. Verify Inference Engine
    # ==========================================
    print("\n[7] Verifying Inference Engine (TTA)...")

    # Inference on the validation set (acting as test set for demo)
    preds = inference_fn(model, val_loader, Config.DEVICE)

    expected_rows = 16  # sample_size used
    assert preds.shape == (
        expected_rows,
        Config.NUM_CLASSES,
    ), f"Prediction shape mismatch: expected ({expected_rows}, {Config.NUM_CLASSES}), got {preds.shape}"

    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions should be probabilities [0, 1]"

    print("    Inference engine: Verified.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
