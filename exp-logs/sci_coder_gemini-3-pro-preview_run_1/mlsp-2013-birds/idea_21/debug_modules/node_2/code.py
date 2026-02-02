import os
import sys
import torch
import torch.nn as nn
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import (
    set_seed,
    get_logger,
    compute_roc_auc,
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import get_dataloaders
from library.model import ManifoldMixupResNet
from library.engine import train_one_epoch, validate, generate_predictions, update_bn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting demonstration of library components...")

    # 1. Setup and Configuration
    # Use debug=True to reduce epochs and dataset size for speed
    config = Config(debug=True)
    set_seed(config.SEED)
    logger = get_logger(os.path.join(config.LOG_DIR, "demo.log"))

    logger.info(f"Configuration initialized: {config}")

    # 2. Dataset and DataLoader Verification
    logger.info("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Verify Train Loader
    try:
        images, targets, rec_ids = next(iter(train_loader))
        logger.info(
            f"Batch loaded. Image shape: {images.shape}, Target shape: {targets.shape}"
        )

        # Assertions
        assert images.shape == (
            config.batch_size,
            3,
            config.IMG_HEIGHT,
            config.IMG_WIDTH,
        ), f"Incorrect image shape: {images.shape}"
        assert targets.shape == (
            config.batch_size,
            config.NUM_CLASSES,
        ), f"Incorrect target shape: {targets.shape}"
        assert len(rec_ids) == config.batch_size, "Incorrect number of rec_ids"
        logger.info("DataLoader verification passed.")
    except StopIteration:
        logger.error("Train loader is empty!")
        raise

    # 3. Model Verification
    logger.info("Initializing Model...")
    device = config.DEVICE
    model = ManifoldMixupResNet(num_classes=config.NUM_CLASSES, pretrained=False).to(
        device
    )

    # Test Standard Forward Pass
    dummy_input = torch.randn(2, 3, config.IMG_HEIGHT, config.IMG_WIDTH).to(device)
    outputs = model(dummy_input)
    assert outputs.shape == (
        2,
        config.NUM_CLASSES,
    ), "Standard forward pass output shape mismatch"

    # Test Mixup Forward Pass
    dummy_target = torch.randint(0, 2, (2, config.NUM_CLASSES)).float().to(device)
    logits, target_a, target_b, lam = model(
        dummy_input, target=dummy_target, mixup=True
    )
    assert logits.shape == (
        2,
        config.NUM_CLASSES,
    ), "Mixup forward pass output shape mismatch"
    assert target_a.shape == dummy_target.shape, "Mixup target_a shape mismatch"
    assert isinstance(lam, float), "Lambda should be a float"

    logger.info("Model verification passed.")

    # 4. Engine Verification (Training and Validation)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Train one epoch (Standard)
    logger.info("Testing training loop (Standard)...")
    loss_std = train_one_epoch(
        model, train_loader, optimizer, device, epoch=0, mixup_active=False
    )
    assert isinstance(loss_std, float), "Train loss should be a float"
    assert loss_std > 0, "Train loss should be positive"
    logger.info(f"Standard training epoch finished. Loss: {loss_std:.4f}")

    # Train one epoch (Mixup)
    logger.info("Testing training loop (Mixup)...")
    loss_mix = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device,
        epoch=1,
        mixup_active=True,
        alpha=config.MIXUP_ALPHA,
    )
    logger.info(f"Mixup training epoch finished. Loss: {loss_mix:.4f}")

    # Validation
    logger.info("Testing validation loop...")
    val_loss, val_auc = validate(model, val_loader, device)
    assert isinstance(val_loss, float), "Val loss should be a float"
    assert isinstance(val_auc, float), "Val AUC should be a float"
    assert 0 <= val_auc <= 1, "AUC must be between 0 and 1"
    logger.info(f"Validation finished. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # 5. SWA Utility Verification
    logger.info("Testing SWA BN Update...")
    # Just running it to ensure no runtime errors
    update_bn(train_loader, model, device)
    logger.info("SWA BN Update finished successfully.")

    # 6. Inference Verification
    logger.info("Testing Inference (Generation of Predictions)...")
    preds = generate_predictions(model, test_loader, device, use_tta=True)

    # Check predictions format
    assert len(preds) > 0, "Predictions dictionary is empty"
    first_key = list(preds.keys())[0]
    first_val = preds[first_key]

    assert isinstance(first_key, int), "Prediction keys should be int (rec_id)"
    assert first_val.shape == (
        config.NUM_CLASSES,
    ), f"Prediction shape mismatch: {first_val.shape}"
    assert np.all(
        (first_val >= 0) & (first_val <= 1)
    ), "Probabilities should be between 0 and 1"
    logger.info(f"Inference verified. Generated predictions for {len(preds)} samples.")

    # 7. Utils Verification
    logger.info("Testing Utilities...")

    # ROC AUC Calculation
    y_true = np.array([[0, 1], [1, 0], [0, 1]])
    y_pred = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    score = compute_roc_auc(y_true, y_pred)
    assert score == 1.0, "ROC AUC calculation logic error"

    # Checkpointing
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "demo_ckpt.pth")
    state = {
        "epoch": 5,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_score": 0.85,
    }
    save_checkpoint(
        state,
        is_best=True,
        checkpoint_dir=config.CHECKPOINT_DIR,
        filename="demo_ckpt.pth",
    )
    assert os.path.exists(ckpt_path), "Checkpoint file not created"

    # Load Checkpoint
    model_new = ManifoldMixupResNet(
        num_classes=config.NUM_CLASSES, pretrained=False
    ).to(device)
    start_epoch, best_score = load_checkpoint(ckpt_path, model_new, device=device)

    assert start_epoch == 5, "Checkpoint loading (epoch) failed"
    assert best_score == 0.85, "Checkpoint loading (best_score) failed"

    # Verify weights loaded
    for p1, p2 in zip(model.parameters(), model_new.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            raise AssertionError("Model weights mismatch after loading checkpoint")

    logger.info("Utilities verification passed.")

    print("\nAll library components verified successfully!")


if __name__ == "__main__":
    main()
