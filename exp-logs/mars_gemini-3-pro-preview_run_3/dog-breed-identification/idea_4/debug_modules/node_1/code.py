import os
import sys
import shutil
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import create_model, freeze_backbone, unfreeze_backbone
from library.engine import train_phase, generate_submission


def run_demo():
    print("============================================================")
    print("  Starting End-to-End Library Demonstration")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Override for Speed & Demo
    # ------------------------------------------------------------------
    # We modify the Config class attributes at runtime to create a
    # lightweight execution environment suitable for a quick demo.

    print("\n[1] Configuring environment for demo run...")

    # Use a separate directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    Config.WORKING_DIR = DEMO_DIR
    Config.OUTPUT_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Use a lightweight model for speed
    Config.MODEL_NAME = "resnet18"

    # Reduce parameters for quick execution
    Config.DEBUG_SAMPLE_SIZE = 50  # Only use 50 images per set
    Config.NUM_WORKERS = 2  # Reduce worker overhead
    BATCH_SIZE = 8
    RESOLUTION = 128  # Low resolution for speed

    # Set fixed seed
    set_seed(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Model: {Config.MODEL_NAME}")
    print(f"    Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # ------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------
    print("\n[2] Initializing DataLoaders (Debug Mode)...")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        resolution=RESOLUTION, batch_size=BATCH_SIZE, debug=True
    )

    # Validation: Check DataLoader outputs
    print("    Verifying DataLoader outputs...")
    images, labels = next(iter(train_loader))

    # Assertions
    assert (
        len(classes) == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} classes, got {len(classes)}"
    assert images.shape == (
        BATCH_SIZE,
        3,
        RESOLUTION,
        RESOLUTION,
    ), f"Expected image shape {(BATCH_SIZE, 3, RESOLUTION, RESOLUTION)}, got {images.shape}"
    assert labels.shape == (
        BATCH_SIZE,
    ), f"Expected label shape {(BATCH_SIZE,)}, got {labels.shape}"

    print(f"    Batch Shape: {images.shape}")
    print(f"    Number of Classes: {len(classes)}")
    print("    Data loading verification passed.")

    # ------------------------------------------------------------------
    # 3. Model Instantiation & Backbone Freezing
    # ------------------------------------------------------------------
    print("\n[3] Creating Model and Freezing Backbone...")

    device = Config.DEVICE
    model = create_model(num_classes=len(classes), pretrained=True)
    model.to(device)

    # Freeze backbone
    freeze_backbone(model)

    # Validation: Verify freezing logic
    # ResNet18 usually has 'fc' as the head.
    # We check if the head is trainable and the first conv layer is frozen.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]

    assert len(trainable_params) > 0, "Model should have trainable parameters (head)"
    assert len(frozen_params) > 0, "Backbone should be frozen"

    print(f"    Trainable parameters: {len(trainable_params)}")
    print(f"    Frozen parameters: {len(frozen_params)}")
    print("    Model freezing verification passed.")

    # ------------------------------------------------------------------
    # 4. Phase 1 Training (Warmup)
    # ------------------------------------------------------------------
    print("\n[4] Running Phase 1 Training (Warmup - 1 Epoch)...")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-2,
    )

    # Run for 1 epoch
    best_loss_p1 = train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,  # No scheduler for 1 epoch demo
        num_epochs=1,
        device=device,
        patience=1,
        save_name="phase1_checkpoint.pth",
    )

    assert os.path.exists(
        os.path.join(Config.OUTPUT_DIR, "phase1_checkpoint.pth")
    ), "Phase 1 checkpoint not saved."
    print(f"    Phase 1 complete. Best Loss: {best_loss_p1:.4f}")

    # ------------------------------------------------------------------
    # 5. Phase 2 Training (Fine-Tuning)
    # ------------------------------------------------------------------
    print("\n[5] Running Phase 2 Training (Fine-Tuning - 1 Epoch)...")

    # Unfreeze backbone
    unfreeze_backbone(model)

    # Verify unfreezing
    frozen_params_check = [p for p in model.parameters() if not p.requires_grad]
    assert (
        len(frozen_params_check) == 0
    ), "All parameters should be trainable after unfreeze."

    # Re-initialize optimizer for all parameters with lower LR
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    best_loss_p2 = train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=1,
        device=device,
        patience=1,
        save_name="demo_best_model.pth",
    )

    assert os.path.exists(
        os.path.join(Config.OUTPUT_DIR, "demo_best_model.pth")
    ), "Best model checkpoint not saved."
    print(f"    Phase 2 complete. Best Loss: {best_loss_p2:.4f}")

    # ------------------------------------------------------------------
    # 6. Submission Generation
    # ------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    # Load best model
    best_model_path = os.path.join(Config.OUTPUT_DIR, "demo_best_model.pth")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    generate_submission(
        model=model,
        test_loader=test_loader,
        classes=classes,
        device=device,
        output_path=Config.SUBMISSION_PATH,
    )

    # Validation: Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission shape: {df_sub.shape}")

    # Check rows (should match debug sample size)
    # Note: test_loader in debug mode has Config.DEBUG_SAMPLE_SIZE items
    # But generate_submission iterates the whole loader.
    expected_rows = min(Config.DEBUG_SAMPLE_SIZE, 1023)  # 1023 is total test size
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    # Check columns (id + 120 breeds)
    assert (
        len(df_sub.columns) == Config.NUM_CLASSES + 1
    ), f"Expected {Config.NUM_CLASSES + 1} columns, got {len(df_sub.columns)}"

    # Check probabilities sum to ~1 (sanity check)
    # Select only breed columns (exclude 'id')
    probs = df_sub.iloc[:, 1:].values
    row_sums = probs.sum(axis=1)
    # Allow small floating point error
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1."

    print("    Submission verification passed.")

    print("\n============================================================")
    print("  Demonstration Completed Successfully")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
