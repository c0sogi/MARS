import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(".")

# Import from provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import MultiTaskModel
from library.engine import train_one_epoch, validate, generate_submission

if __name__ == "__main__":
    print("--- Starting Demonstration Script ---")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("[1] Configuring environment...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.IMG_SIZE = 256  # Reduced size for faster processing
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("[2] Initializing DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        img_size=Config.IMG_SIZE,
        debug=True,
        load_cached_data=False,
    )

    # Fetch a single batch to verify shapes
    images, targets, masks = next(iter(train_loader))

    print(f"    Train Batch - Images: {images.shape}")
    print(f"    Train Batch - Targets: {targets.shape}")
    print(f"    Train Batch - Masks: {masks.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Image shape mismatch"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Target shape mismatch"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Mask shape mismatch"
    assert targets.dtype == torch.float32, "Targets should be float32"

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("[3] Initializing Model and running forward pass...")

    # Use a lightweight backbone (resnet18) and no pretraining for speed
    model = MultiTaskModel(backbone_name="resnet18", pretrained=False)
    model.to(device)

    # Move batch to device
    images = images.to(device)

    # Forward pass
    cls_logits, seg_logits = model(images)

    print(f"    Output - Cls Logits: {cls_logits.shape}")
    print(f"    Output - Seg Logits: {seg_logits.shape}")

    # Assertions
    assert cls_logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Classification output shape mismatch"
    assert seg_logits.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Segmentation output shape mismatch"

    # -------------------------------------------------------------------------
    # 4. Training Loop (Single Epoch)
    # -------------------------------------------------------------------------
    print("[4] Testing Training Loop...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Initialize scaler (enabled only if cuda is available)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Run one epoch
    loss, cls_loss, seg_loss = train_one_epoch(
        model, train_loader, optimizer, scaler, device
    )

    print(
        f"    Epoch Result: Total Loss={loss:.4f}, Cls Loss={cls_loss:.4f}, Seg Loss={seg_loss:.4f}"
    )

    # Assertions
    assert not np.isnan(loss), "Training loss is NaN"
    assert loss > 0, "Training loss should be positive"

    # -------------------------------------------------------------------------
    # 5. Validation
    # -------------------------------------------------------------------------
    print("[5] Testing Validation...")

    val_score = validate(model, val_loader, device)
    print(f"    Validation AUC: {val_score:.4f}")

    # Assertions
    assert 0.0 <= val_score <= 1.0, "Validation AUC out of range [0, 1]"

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("[6] Testing Inference and Submission Generation...")

    # Save the current model as 'best_model.pth' so generate_submission can load it
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"    Saved dummy best model to {Config.BEST_MODEL_PATH}")

    generate_submission(model, test_loader, device)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission File Shape: {df_sub.shape}")

    # Assertions
    assert "StudyInstanceUID" in df_sub.columns, "StudyInstanceUID column missing"
    assert (
        len(df_sub.columns) == Config.NUM_CLASSES + 1
    ), "Incorrect number of columns in submission"

    # Check if StudyInstanceUIDs match the test loader
    # Note: In debug mode, test_loader is a subset
    test_uids_count = len(test_loader.dataset)
    assert (
        len(df_sub) == test_uids_count
    ), f"Submission row count ({len(df_sub)}) does not match test dataset size ({test_uids_count})"

    print("\n--- Demonstration Completed Successfully ---")
