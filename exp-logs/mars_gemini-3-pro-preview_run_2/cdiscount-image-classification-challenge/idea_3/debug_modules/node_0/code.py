import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.dataset import (
    BSONDataset,
    collate_flatten,
    collate_product,
    get_category_mapping,
)
from library.model import get_model
from library.utils import LabelSmoothingCrossEntropy, Mixup, accuracy
from library.engine import fit


def run_demo():
    # 1. Setup Configuration for Demo
    print("Setting up configuration for demonstration...")
    Config.seed_everything(42)

    # Override Config for speed and demonstration purposes
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2
    Config.DEBUG = True
    # Disable pretraining to avoid downloading weights during demo
    Config.PRETRAINED = False

    # We will use a very small subset for the demo to ensure < 1 hour runtime
    DEMO_TRAIN_SIZE = 200
    DEMO_VAL_SIZE = 50
    DEMO_TEST_SIZE = 50

    print(f"Device: {Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Validate Dataset and DataLoader
    print("\nInitializing Datasets...")

    # Train Dataset
    train_ds = BSONDataset(
        metadata_csv=Config.TRAIN_META,
        bson_file=Config.TRAIN_BSON,
        split="train",
        debug_size=DEMO_TRAIN_SIZE,
    )

    # Val Dataset
    val_ds = BSONDataset(
        metadata_csv=Config.VAL_META,
        bson_file=Config.TRAIN_BSON,  # Validation data is physically in train.bson
        split="val",
        debug_size=DEMO_VAL_SIZE,
    )

    # Test Dataset
    test_ds = BSONDataset(
        metadata_csv=Config.TEST_META,
        bson_file=Config.TEST_BSON,
        split="test",
        debug_size=DEMO_TEST_SIZE,
    )

    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")
    print(f"Test size: {len(test_ds)}")

    # Verify __getitem__ structure
    print("Verifying dataset item structure...")
    img_list, label, pid = train_ds[0]

    # Assertions for Dataset
    if not isinstance(img_list, list):
        raise AssertionError("Dataset should return a list of images")
    if len(img_list) == 0:
        raise AssertionError("Product should have at least one image")
    if not isinstance(img_list[0], torch.Tensor):
        raise AssertionError("Image should be a tensor")
    # Check image shape: (3, H, W)
    if img_list[0].shape != (3, Config.INPUT_SIZE, Config.INPUT_SIZE):
        raise AssertionError(
            f"Image shape mismatch. Expected {(3, Config.INPUT_SIZE, Config.INPUT_SIZE)}, Got {img_list[0].shape}"
        )
    if not isinstance(label, int):
        raise AssertionError("Label should be an int")

    # Initialize DataLoaders
    print("\nInitializing DataLoaders...")
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_flatten,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_product,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_product,
        pin_memory=Config.PIN_MEMORY,
    )

    # Verify Train Batch (Flattened)
    print("Verifying Train Batch...")
    images, targets = next(iter(train_loader))
    print(f"Train Batch Shapes - Images: {images.shape}, Targets: {targets.shape}")

    if images.dim() != 4:
        raise AssertionError("Train images should be 4D (N, C, H, W)")
    if targets.dim() != 1:
        raise AssertionError("Train targets should be 1D (N,)")

    # Verify Val Batch (Product-grouped)
    print("Verifying Val Batch...")
    v_images, v_pids, v_labels, v_sizes = next(iter(val_loader))
    print(
        f"Val Batch Shapes - Images: {v_images.shape}, Pids: {v_pids.shape}, Sizes: {v_sizes.shape}"
    )

    if v_images.dim() != 4:
        raise AssertionError("Val images should be 4D")
    if v_sizes.sum() != v_images.shape[0]:
        raise AssertionError("Sum of sizes should equal total images in batch")

    # 3. Validate Model
    print("\nInitializing Model...")
    # Force pretrained=False to ensure no network calls
    model = get_model(pretrained=False, num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)

    # Forward pass check
    print("Running dummy forward pass...")
    dummy_input = torch.randn(2, 3, Config.INPUT_SIZE, Config.INPUT_SIZE).to(
        Config.DEVICE
    )
    with torch.no_grad():
        output = model(dummy_input)

    if output.shape != (2, Config.NUM_CLASSES):
        raise AssertionError(
            f"Model output shape mismatch. Expected {(2, Config.NUM_CLASSES)}, Got {output.shape}"
        )
    print("Model forward pass successful.")

    # 4. Validate Components
    print("\nValidating Components...")

    # Loss
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    dummy_target = torch.tensor([0, 1]).to(Config.DEVICE)
    loss = criterion(output, dummy_target)
    if torch.isnan(loss):
        raise AssertionError("Loss is NaN")
    print(f"Loss calculation successful: {loss.item():.4f}")

    # Mixup
    mixup = Mixup(alpha=0.2)
    m_imgs, m_ya, m_yb, lam = mixup(dummy_input, dummy_target)
    if m_imgs.shape != dummy_input.shape:
        raise AssertionError("Mixup image shape mismatch")
    print("Mixup successful.")

    # Accuracy
    acc = accuracy(output, dummy_target, topk=(1,))
    print(f"Accuracy calculation successful: {acc[0].item()}")

    # 5. Run Training Loop (Fit)
    print("\nStarting Training Loop (1 Epoch, Debug subset)...")
    # This function handles training, validation, checkpointing, and inference
    fit(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=Config.EPOCHS,
        device=Config.DEVICE,
    )

    # 6. Verify Outputs
    print("\nVerifying Outputs...")

    # Checkpoints
    if not os.path.exists(Config.CHECKPOINT_DIR):
        raise AssertionError("Checkpoint directory missing")

    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "checkpoint.pth.tar")
    if not os.path.exists(ckpt_path):
        raise AssertionError("Checkpoint file missing")

    # Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file missing")

    # Read submission
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")

    if df_sub.shape[0] != DEMO_TEST_SIZE:
        raise AssertionError(
            f"Submission rows mismatch. Expected {DEMO_TEST_SIZE}, got {df_sub.shape[0]}"
        )

    if "_id" not in df_sub.columns or "category_id" not in df_sub.columns:
        raise AssertionError("Submission columns mismatch")

    print("\nAll demonstrations and validations passed successfully!")


if __name__ == "__main__":
    run_demo()
