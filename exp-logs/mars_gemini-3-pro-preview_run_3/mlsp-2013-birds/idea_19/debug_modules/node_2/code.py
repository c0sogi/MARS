import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_robust_roc_auc,
    save_checkpoint,
    load_checkpoint,
)
from library.data import get_fold_loaders, get_test_loader
from library.models import get_model
from library.sam import SAM
from library.engine import train_one_epoch, evaluate


def run_demo():
    print("==== Starting Library Demo ====")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration and Random Seeds...")

    # Override Config for speed (Demo Mode)
    Config.DEBUG_MAX_SAMPLES = 60  # Small subset for quick execution
    Config.BATCH_SIZE = 16  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.NUM_FOLDS = 2  # Reduce folds for stratification check

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration updated for fast demonstration.")

    # -------------------------------------------------------------------------
    # 2. Data Loading (library.data)
    # -------------------------------------------------------------------------
    print("\n[2] Initializing Data Loaders...")

    # Get training and validation loaders for Fold 0
    # We force reload_cached_data=False to demonstrate raw loading logic at least once,
    # or rely on the cache logic if it works. Given the short runtime requirement,
    # we'll let it handle caching but since we changed DEBUG_MAX_SAMPLES, we should
    # probably clear old cache files if they exist to ensure the limit applies.
    # However, the library saves to ./working/idea_19. We can just run it.
    # To be safe regarding the debug limit, we'll rely on the library logic.

    # Note: prepare_data checks cache. If cache exists from a full run, it might load that.
    # For this demo to be fast, we assume the environment is clean or we accept loading cached data.
    # Given the instructions, we can't delete pre-existing files easily if they aren't ours,
    # but we can assume ./working is ours to manage.

    train_loader, val_loader = get_fold_loaders(fold_idx=0, load_cached_data=False)
    test_loader = get_test_loader(load_cached_data=False)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches: {len(val_loader)}")

    # Verify Data Shapes
    images, labels, ids = next(iter(train_loader))
    print(
        f"    Batch Image Shape: {images.shape} (Expected: [{Config.BATCH_SIZE}, 3, 224, 224])"
    )
    print(
        f"    Batch Label Shape: {labels.shape} (Expected: [{Config.BATCH_SIZE}, 19])"
    )

    assert images.shape == (Config.BATCH_SIZE, 3, 224, 224), "Incorrect Image Shape"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect Label Shape"
    assert images.dtype == torch.float32, "Images should be float32"

    # -------------------------------------------------------------------------
    # 3. Model Initialization (library.models)
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Model (ResNet18)...")

    device = Config.DEVICE
    model = get_model(
        "resnet18", num_classes=Config.NUM_CLASSES, pretrained=False
    )  # False for speed
    model.to(device)

    # Verify output layer
    assert model.fc.out_features == Config.NUM_CLASSES, "Model output classes mismatch"
    print("    Model initialized and moved to device.")

    # -------------------------------------------------------------------------
    # 4. Optimizer Setup (library.sam)
    # -------------------------------------------------------------------------
    print("\n[4] Setting up SAM Optimizer...")

    base_optimizer = torch.optim.AdamW
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        lr=Config.LEARNING_RATE,
        rho=Config.SAM_RHO,
        weight_decay=Config.WEIGHT_DECAY,
    )
    print("    SAM Optimizer initialized.")

    # -------------------------------------------------------------------------
    # 5. Training Loop (library.engine)
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch)...")

    criterion = nn.BCEWithLogitsLoss()

    # Train
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"    Training Loss: {train_loss:.4f}")

    # Evaluate
    val_loss, val_auc = evaluate(model, val_loader, criterion, device)
    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Validation ROC AUC: {val_auc:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert 0.0 <= val_auc <= 1.0, "AUC score out of range"

    # -------------------------------------------------------------------------
    # 6. Checkpointing (library.utils)
    # -------------------------------------------------------------------------
    print("\n[6] Testing Checkpoint System...")

    ckpt_filename = "demo_checkpoint.pth"
    save_checkpoint(model, optimizer, epoch=1, score=val_auc, filename=ckpt_filename)

    # Create a new model instance to verify loading
    new_model = get_model("resnet18", num_classes=Config.NUM_CLASSES, pretrained=False)
    new_model.to(device)

    # Load
    checkpoint = load_checkpoint(new_model, None, ckpt_filename, device=device)

    assert checkpoint["epoch"] == 1, "Incorrect epoch in checkpoint"
    assert checkpoint["score"] == val_auc, "Incorrect score in checkpoint"

    # Verify weights match (simple check on one parameter)
    orig_param = list(model.parameters())[0]
    loaded_param = list(new_model.parameters())[0]
    assert torch.equal(orig_param, loaded_param), "Model weights did not load correctly"

    print(
        f"    Checkpoint saved to {os.path.join(Config.WORKING_DIR, ckpt_filename)} and verified."
    )

    # -------------------------------------------------------------------------
    # 7. Utility Verification (library.utils)
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Robust ROC AUC Calculation...")

    # Create dummy data where one class is missing in ground truth
    y_true = np.array(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]
    )  # Class 2 is all zeros
    y_pred = np.array(
        [[0.9, 0.1, 0.2], [0.8, 0.2, 0.3], [0.2, 0.8, 0.1], [0.1, 0.9, 0.4]]
    )

    # Standard AUC would fail or warn for class 2. Robust AUC should skip it.
    # Class 0 AUC: Perfect (1.0)
    # Class 1 AUC: Perfect (1.0)
    # Class 2: Skipped
    # Average: 1.0

    score = calculate_robust_roc_auc(y_true, y_pred)
    print(f"    Robust AUC Score: {score:.4f}")
    assert score == 1.0, "Robust AUC calculation failed logic check"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
