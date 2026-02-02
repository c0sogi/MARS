import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_macro_f1
from library.dataset import get_dataloaders
from library.model import HerbariumResNet
from library.trainer import Trainer


def run_demo():
    print("=== Starting Herbarium Classification Demo ===")

    # ------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------
    # Override Config for a fast demonstration
    print("[Setup] Overriding configuration for fast demo execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Small sample for speed
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_EPOCHS = 1  # Only 1 epoch for demo
    Config.NUM_WORKERS = 2  # Reduce workers for small data

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Define a temporary working directory for this demo to avoid conflicts
    demo_working_dir = os.path.join(Config.WORKING_DIR, "demo_run")
    os.makedirs(demo_working_dir, exist_ok=True)
    Config.MODEL_SAVE_PATH = os.path.join(demo_working_dir, "model_demo.pth")
    # Disable loading cached mapping to ensure mapping matches our debug subset
    load_cached_mapping = False

    print(f"[Setup] Device: {Config.DEVICE}")
    print(f"[Setup] Debug Mode: {Config.DEBUG}")

    # ------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------
    print("\n[Data] Initializing DataLoaders...")

    # We pass load_cached_data=False so the class mapping is generated
    # specifically for our random debug subset, ensuring model output size matches labels.
    loaders = get_dataloaders(
        train_csv_path=Config.TRAIN_CSV,
        val_csv_path=Config.VAL_CSV,
        test_csv_path=Config.TEST_CSV,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=Config.DEBUG,
        load_cached_data=load_cached_mapping,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    num_classes = loaders["num_classes"]

    print(f"[Data] Number of classes in debug subset: {num_classes}")
    print(f"[Data] Train batches: {len(train_loader)}")
    print(f"[Data] Val batches: {len(val_loader)}")

    # Validation: Check batch structure
    images, labels = next(iter(train_loader))
    print(f"[Data] Sample Batch Shape - Images: {images.shape}, Labels: {labels.shape}")

    assert images.dim() == 4, "Images must be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images must have 3 channels"
    assert labels.dim() == 1, "Labels must be 1D tensors"
    assert images.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"

    # ------------------------------------------------------------------
    # 3. Model Initialization
    # ------------------------------------------------------------------
    print("\n[Model] Initializing HerbariumResNet...")
    model = HerbariumResNet(num_classes=num_classes, pretrained=True)
    model = model.to(Config.DEVICE)

    # Validation: Check forward pass
    with torch.no_grad():
        dummy_input = images.to(Config.DEVICE)
        dummy_output = model(dummy_input)
        print(f"[Model] Output shape: {dummy_output.shape}")
        assert dummy_output.shape == (
            Config.BATCH_SIZE,
            num_classes,
        ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, num_classes)}, got {dummy_output.shape}"

    # ------------------------------------------------------------------
    # 4. Training Setup
    # ------------------------------------------------------------------
    print("\n[Training] Setting up Optimizer and Scheduler...")
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        patience=Config.PATIENCE,
    )

    # ------------------------------------------------------------------
    # 5. Execution (Fit)
    # ------------------------------------------------------------------
    print("\n[Training] Starting Training Loop...")
    # This runs training and validation for the specified epochs
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # Validation: Check if model file was created
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"[Training] Success: Best model saved to {Config.MODEL_SAVE_PATH}")
    else:
        # It's possible no improvement happened if F1 stayed 0, but usually it saves at least once if logic allows.
        # In this short demo with random weights/data, validation F1 might be 0.
        print(
            "[Training] Note: No model saved (likely no F1 improvement over 0.0 in 1 epoch), which is acceptable for a random debug run."
        )

    # ------------------------------------------------------------------
    # 6. Inference / Submission Check
    # ------------------------------------------------------------------
    print("\n[Inference] Testing prediction on Test Loader...")
    model.eval()
    test_images, test_ids = next(iter(test_loader))
    test_images = test_images.to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(test_images)
        _, preds = torch.max(outputs, 1)

    print(f"[Inference] Test IDs: {test_ids.tolist()[:5]}...")
    print(f"[Inference] Predictions (Class Indices): {preds.tolist()[:5]}...")

    assert len(preds) == len(test_ids), "Number of predictions must match number of IDs"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
