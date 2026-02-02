import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import from provided library
from library.config import Config
from library.utils import seed_everything, save_clean_checkpoint
from library.data import get_dataloaders
from library.model_factory import create_model, get_llrd_params
from library.trainer import train_one_epoch, validate, update_swa
from library.calibration import TemperatureScaler


def demo_pipeline():
    print("Starting Library Demonstration...")

    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print("\n[1] Configuring environment for demo...")

    # Set reproducible seed
    seed_everything(42)

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 60  # Small sample for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.WORK_DIR = "./working/demo_run"
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    print(f"    Device: {Config.DEVICE}")
    print(f"    Work Dir: {Config.WORK_DIR}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n[2] Testing Data Loading...")

    # Force reload of class mapping to ensure it writes to our new WORK_DIR
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        load_cached_data=False
    )

    # Assertions
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"
    assert len(class_to_idx) > 0, "Class mapping is empty"

    # Check batch structure
    images, targets = next(iter(train_loader))
    assert images.dim() == 4, f"Expected 4D image tensor, got {images.dim()}"
    assert targets.dim() == 1, f"Expected 1D target tensor, got {targets.dim()}"
    assert (
        images.size(0) == Config.BATCH_SIZE
    ), f"Expected batch size {Config.BATCH_SIZE}, got {images.size(0)}"

    print(f"    Data loaded successfully. Classes: {len(class_to_idx)}")

    # ==========================================
    # 3. Model Factory & LLRD
    # ==========================================
    print("\n[3] Testing Model Creation and LLRD...")

    model_name = Config.MODELS[0]  # Use the first model (ConvNeXt)
    print(f"    Creating model: {model_name}")

    model = create_model(model_name, num_classes=len(class_to_idx))
    model.to(Config.DEVICE)

    assert isinstance(model, nn.Module), "Model is not a torch.nn.Module"

    # Test LLRD Parameter Grouping
    lr = 1e-3
    param_groups = get_llrd_params(model, model_name, lr=lr)

    assert len(param_groups) > 0, "No parameter groups generated"
    assert "lr" in param_groups[0], "Parameter group missing 'lr' key"
    assert (
        "weight_decay" in param_groups[0]
    ), "Parameter group missing 'weight_decay' key"

    # Verify different LRs are assigned (Head should have higher LR than backbone)
    lrs = [g["lr"] for g in param_groups]
    assert len(set(lrs)) > 1, "LLRD did not assign different learning rates"

    print("    Model created and parameters grouped successfully.")

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print("\n[4] Testing Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(param_groups)
    criterion = nn.CrossEntropyLoss()

    # Train one epoch
    avg_loss = train_one_epoch(
        epoch=1,
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=Config.DEVICE,
    )

    assert isinstance(avg_loss, float), "Train loss is not a float"
    assert avg_loss > 0, "Train loss should be positive"

    print(f"    Training complete. Loss: {avg_loss:.4f}")

    # ==========================================
    # 5. Validation
    # ==========================================
    print("\n[5] Testing Validation...")

    val_loss, val_acc, val_logits, val_labels = validate(
        model=model, loader=val_loader, criterion=criterion, device=Config.DEVICE
    )

    assert val_logits.size(0) == val_labels.size(
        0
    ), "Mismatch between logits and labels count"
    assert val_logits.size(1) == len(
        class_to_idx
    ), "Logits dimension does not match num_classes"

    print(f"    Validation complete. Acc: {val_acc:.2f}%")

    # ==========================================
    # 6. Calibration
    # ==========================================
    print("\n[6] Testing Temperature Scaling Calibration...")

    scaler = TemperatureScaler(device=Config.DEVICE)

    # Fit scaler
    scaler.fit(val_logits, val_labels)

    # Check if temperature was updated (initial is 1.5)
    temp = scaler.get_temperature()
    print(f"    Optimized Temperature: {temp:.4f}")

    # Predict probabilities
    probs = scaler.predict_proba(val_logits)

    # Verify probabilities sum to 1
    sums = probs.sum(dim=1)
    ones = torch.ones_like(sums)
    assert torch.allclose(sums, ones, atol=1e-4), "Probabilities do not sum to 1"

    print("    Calibration successful.")

    # ==========================================
    # 7. SWA Integration
    # ==========================================
    print("\n[7] Testing SWA Integration...")

    swa_model = torch.optim.swa_utils.AveragedModel(model)

    # Mocking an epoch that triggers SWA (Config.SWA_START_EPOCH is usually 24)
    # We force the update by passing an epoch >= SWA_START_EPOCH
    mock_epoch = Config.SWA_START_EPOCH + 1

    # Capture state before update
    # Note: AveragedModel is lazy init, so we need to run a forward pass or update once to init
    # But update_parameters handles initialization.

    try:
        update_swa(swa_model, model, swa_scheduler=None, epoch=mock_epoch)
        print("    SWA update executed successfully.")
    except Exception as e:
        raise AssertionError(f"SWA update failed: {e}")

    # ==========================================
    # 8. Utils Verification
    # ==========================================
    print("\n[8] Testing Utils...")

    ckpt_path = os.path.join(Config.WORK_DIR, "test_ckpt.pth")
    save_clean_checkpoint(swa_model, ckpt_path)
    assert os.path.exists(ckpt_path), "Checkpoint file was not saved"
    print("    Checkpoint saved successfully.")

    print("\n" + "=" * 40)
    print("ALL DEMONSTRATION STEPS PASSED")
    print("=" * 40)


if __name__ == "__main__":
    demo_pipeline()
