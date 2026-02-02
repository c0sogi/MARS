import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_loaders, get_test_loader
from library.model import CassavaClassifier
from library.loss import SoftTargetCrossEntropy
from library.engine import get_optimizer_llrd, train_one_epoch, validate


def run_demo():
    # 1. Setup & Configuration
    print("--- 1. Setup & Configuration ---")
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Initialize Config with debug=True for small subset and few epochs
    config = Config(debug=True)

    # Override model to a smaller one for speed in this demonstration
    config.model_name = "resnet18"
    config.batch_size_coarse = 16

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)
    print("Configuration initialized (Debug Mode).")

    # 2. Data Loading Demonstration
    print("\n--- 2. Data Loading Demonstration ---")
    # Get loaders for the 'coarse' phase
    train_loader, val_loader, mixup_fn = get_loaders(config, phase="coarse")

    print(f"Train Loader length: {len(train_loader)}")
    print(f"Val Loader length: {len(val_loader)}")

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Verify Image shape: (B, 3, H, W)
    assert len(images.shape) == 4, "Images should be 4D tensors"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert (
        images.shape[2] == config.img_size_coarse
    ), f"Height should be {config.img_size_coarse}"

    # Verify Mixup functionality
    if mixup_fn is not None:
        print("Testing Mixup function...")
        mixed_images, mixed_targets = mixup_fn(images.to(device), labels.to(device))
        print(f"Mixed Targets Shape: {mixed_targets.shape}")

        # Mixed targets should be (B, Num_Classes) (One-hot/Soft)
        assert mixed_targets.shape == (
            images.shape[0],
            config.num_classes,
        ), "Mixup targets should be shape (Batch, Num_Classes)"
        assert torch.is_floating_point(mixed_targets), "Mixed targets should be float"

    # 3. Model Initialization
    print("\n--- 3. Model Initialization ---")
    model = CassavaClassifier(config, pretrained=True)
    model.to(device)

    # Dummy forward pass
    print("Running dummy forward pass...")
    with torch.no_grad():
        dummy_input = torch.randn(
            2, 3, config.img_size_coarse, config.img_size_coarse
        ).to(device)
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        2,
        config.num_classes,
    ), f"Model output shape mismatch. Expected (2, {config.num_classes})"

    # 4. Training Components
    print("\n--- 4. Training Components ---")
    # Loss
    criterion = SoftTargetCrossEntropy()

    # Optimizer (LLRD)
    optimizer = get_optimizer_llrd(model, config, learning_rate=1e-3)
    print(f"Optimizer created with {len(optimizer.param_groups)} param groups.")

    # Check if param groups have different LRs (simple check for LLRD logic)
    if len(optimizer.param_groups) > 1:
        lr_0 = optimizer.param_groups[0]["lr"]
        lr_last = optimizer.param_groups[-1]["lr"]
        print(f"LR Group 0: {lr_0:.2e}, LR Group Last: {lr_last:.2e}")

    # Scaler for Mixed Precision
    scaler = torch.amp.GradScaler("cuda")

    # 5. Execution Loop (Engine)
    print("\n--- 5. Execution Loop (Train & Validate) ---")
    # Train for 1 epoch (debug mode sets epochs to 1 usually, but we call explicitly)
    print("Starting training epoch...")
    avg_loss = train_one_epoch(
        epoch=1,
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        scaler=scaler,
        model_ema=None,  # Skipping EMA for simple demo
        mixup_fn=mixup_fn,
        grad_accum_steps=config.grad_accum_steps_coarse,
    )
    print(f"Training Epoch Completed. Avg Loss: {avg_loss:.4f}")

    # Validate
    print("Starting validation...")
    val_loss, val_acc = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        config=config,
    )
    print(f"Validation Completed. Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # 6. Inference Demonstration
    print("\n--- 6. Inference Demonstration ---")
    test_loader = get_test_loader(config, phase="coarse")
    print(f"Test Loader length: {len(test_loader)}")

    # Run inference on one batch
    model.eval()
    with torch.no_grad():
        for test_images, _ in test_loader:  # test loader returns dummy labels
            test_images = test_images.to(device)
            outputs = model(test_images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            print(f"Inference Batch Output Shape: {outputs.shape}")
            print(f"Sample Predictions: {preds[:5].cpu().tolist()}")

            assert outputs.shape[1] == config.num_classes
            break  # Just one batch

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
