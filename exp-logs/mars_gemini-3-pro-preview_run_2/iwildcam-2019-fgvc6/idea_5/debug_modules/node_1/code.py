import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Ensure the current directory is in the python path to correctly import 'library'
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.dataset import create_dataloaders
from library.model import EfficientNetB4Native
from library.engine import (
    train_one_epoch,
    evaluate,
    configure_model_for_stage,
    get_optimizer,
    get_scheduler,
)


def main():
    print("=== Starting Animal Classification Pipeline Demo ===")

    # 1. Setup and Configuration Override
    # We modify the Config class directly to optimize for a quick demonstration run.
    print("[Setup] Configuring environment...")
    seed_everything(42)

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Use a tiny subset for speed
    Config.BATCH_SIZE = 8  # Small batch size for demonstration
    Config.NUM_WORKERS = 2  # Reduce workers to avoid overhead on small data
    Config.EPOCHS_STAGE1 = 1  # Run only 1 epoch
    Config.EPOCHS_STAGE2 = 1  # Run only 1 epoch

    # Ensure working directory exists (as per Config)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading Verification
    print("\n[Data] Creating DataLoaders...")
    # Disable caching to ensure we test the raw loading logic
    train_loader, val_loader, test_loader = create_dataloaders(load_cached_data=False)

    # Verify DataLoader properties
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    print(f"Train Loader length: {len(train_loader)} batches")

    # Fetch a single batch to verify shapes
    try:
        images, targets = next(iter(train_loader))
        print(f"Sample Batch - Images: {images.shape}, Targets: {targets.shape}")

        # Validation: Check shapes match Config
        expected_img_shape = (
            Config.BATCH_SIZE,
            3,
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        )
        expected_target_shape = (Config.BATCH_SIZE,)

        if images.shape != expected_img_shape:
            raise AssertionError(
                f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
            )
        if targets.shape != expected_target_shape:
            raise AssertionError(
                f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"
            )

        print("[Data] Data shapes verified successfully.")

    except StopIteration:
        raise RuntimeError(
            "DataLoader is empty! Check dataset paths or debug sample size."
        )

    # 3. Model Initialization and Forward Pass
    print("\n[Model] Instantiating EfficientNetB4Native...")
    device = torch.device(Config.DEVICE)
    model = EfficientNetB4Native().to(device)

    # Verify Forward Pass
    print("[Model] Running dummy forward pass...")
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)

    print(f"Output Logits Shape: {outputs.shape}")
    if outputs.shape != (Config.BATCH_SIZE, Config.NUM_CLASSES):
        raise AssertionError(
            f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"
        )

    criterion = nn.CrossEntropyLoss()

    # 4. Stage 1 Training: Head Alignment (Frozen Backbone)
    print("\n[Training] Starting Stage 1: Head Alignment...")
    configure_model_for_stage(model, stage=1)

    # Verify Freezing Logic
    # The first layer of the backbone should be frozen
    first_backbone_param = next(model.backbone.parameters())
    if first_backbone_param.requires_grad:
        raise AssertionError(
            "Stage 1 Error: Backbone parameters should be frozen (requires_grad=False)."
        )

    # The classifier should be trainable
    if not model.classifier.weight.requires_grad:
        raise AssertionError(
            "Stage 1 Error: Classifier parameters should be trainable."
        )
    print("[Training] Stage 1 freezing logic verified.")

    optimizer = get_optimizer(model, stage=1)

    # Train one epoch
    loss_stage1 = train_one_epoch(model, train_loader, optimizer, device, criterion)
    print(f"[Training] Stage 1 Epoch Loss: {loss_stage1:.4f}")

    # Evaluate
    val_loss, val_f1 = evaluate(model, val_loader, device, criterion)
    print(f"[Evaluation] Validation F1 Score: {val_f1:.4f}")

    # 5. Stage 2 Training: Fine-Tuning (Unfrozen Top Layers)
    print("\n[Training] Starting Stage 2: Fine-Tuning...")
    configure_model_for_stage(model, stage=2)

    # Verify Unfreezing Logic
    # The last block of the backbone should now be trainable
    # efficientnet_b4.features is a Sequential container
    last_block_params = list(model.backbone[-1].parameters())
    if not last_block_params[0].requires_grad:
        raise AssertionError("Stage 2 Error: Top backbone layers should be unfrozen.")

    # The first layer should STILL be frozen
    first_backbone_param = next(model.backbone[0].parameters())
    if first_backbone_param.requires_grad:
        raise AssertionError(
            "Stage 2 Error: Bottom backbone layers should remain frozen."
        )
    print("[Training] Stage 2 unfreezing logic verified.")

    optimizer = get_optimizer(model, stage=2)
    scheduler = get_scheduler(optimizer, epochs=Config.EPOCHS_STAGE2)

    loss_stage2 = train_one_epoch(model, train_loader, optimizer, device, criterion)
    scheduler.step()
    print(f"[Training] Stage 2 Epoch Loss: {loss_stage2:.4f}")

    # 6. Inference Demonstration
    print("\n[Inference] Generating predictions on Test Set (Subset)...")
    model.eval()
    test_ids = []
    test_preds = []

    # We iterate through the test loader (which is small due to DEBUG mode)
    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(test_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            _, preds = torch.max(logits, 1)

            test_preds.extend(preds.cpu().numpy())

            # In a real scenario, we would track IDs.
            # Here we just demonstrate the prediction generation.

    print(f"[Inference] Generated {len(test_preds)} predictions.")

    # Create a dummy submission dataframe to show format
    # Note: In the real test_loader, we would need to track IDs properly.
    # Since dataset.py returns (image, label), we rely on the order matching the metadata.
    # The metadata was loaded in create_dataloaders.

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
