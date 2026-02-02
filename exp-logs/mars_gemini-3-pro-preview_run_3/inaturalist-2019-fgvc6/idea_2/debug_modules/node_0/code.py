import os
import torch
import pandas as pd
import numpy as np
import logging
from library import config, utils, dataset, model, engine


def run_demonstration():
    print("=== Starting Library Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Setup and Utils Verification
    # -------------------------------------------------------------------------
    print("\n[Step 1] Verifying Utils...")

    # Verify seed setting
    utils.seed_everything(42)
    print("  Seed set successfully.")

    # Verify device detection
    device = utils.get_device()
    print(f"  Detected device: {device}")

    # Verify logger
    logger = utils.get_logger("demo_logger")
    logger.info("  Logger initialized and working.")

    # -------------------------------------------------------------------------
    # 2. Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print("\n[Step 2] Configuring for fast demonstration...")

    # Override global config for speed
    config.DEBUG_SAMPLE_SIZE = 50  # Only use 50 images
    config.NUM_WORKERS = 2  # Reduce workers for small batch

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    print(f"  Debug sample size set to: {config.DEBUG_SAMPLE_SIZE}")
    print(f"  Working directory: {config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 3. Dataset and DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Dataset and DataLoader...")

    # Load metadata
    train_df, val_df, test_df = dataset.load_metadata(
        debug_size=config.DEBUG_SAMPLE_SIZE
    )

    # Assertions to verify data loading
    assert len(train_df) == config.DEBUG_SAMPLE_SIZE, "Train DF size mismatch"
    assert len(val_df) == config.DEBUG_SAMPLE_SIZE, "Val DF size mismatch"
    assert len(test_df) == config.DEBUG_SAMPLE_SIZE, "Test DF size mismatch"
    print("  Metadata loaded successfully.")

    # Create DataLoaders
    batch_size = 4
    image_size = 224

    train_loader = dataset.get_dataloader(
        train_df,
        image_size=image_size,
        batch_size=batch_size,
        is_training=True,
        sampling_strategy="instance_balanced",
    )

    val_loader = dataset.get_dataloader(
        val_df, image_size=image_size, batch_size=batch_size, is_training=False
    )

    # Verify Batch Structure
    images, targets, image_ids = next(iter(train_loader))

    print(f"  Batch shapes - Images: {images.shape}, Targets: {targets.shape}")

    # Assertions for tensor shapes
    assert images.shape == (
        batch_size,
        3,
        image_size,
        image_size,
    ), "Incorrect image batch shape"
    assert targets.shape == (batch_size,), "Incorrect target batch shape"
    assert images.dtype == torch.float32, "Images should be float32"

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Model creation...")

    # Create model (pretrained=False for speed/offline safety in demo)
    net = model.create_model(num_classes=config.NUM_CLASSES, pretrained=False)
    net = net.to(device)

    # Verify model type
    assert isinstance(net, torch.nn.Module), "Model is not a torch.nn.Module"
    print(f"  Model {config.MODEL_NAME} created successfully.")

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = images.to(device)
        outputs = net(dummy_input)

    print(f"  Output shape: {outputs.shape}")
    assert outputs.shape == (
        batch_size,
        config.NUM_CLASSES,
    ), "Model output shape mismatch"

    # Verify Freezing/Unfreezing Logic
    print("  Testing backbone freezing logic...")
    model.set_backbone_trainable(net, trainable=False)
    # Check if a backbone parameter is frozen (requires_grad=False)
    # EfficientNetV2 usually starts with 'conv_stem' or similar
    param_example = next(net.parameters())
    assert param_example.requires_grad is False, "Backbone should be frozen"

    # Check if classifier is unfrozen (requires_grad=True)
    # The classifier head in timm efficientnet is usually 'classifier'
    classifier_params = list(net.classifier.parameters())
    if classifier_params:
        assert (
            classifier_params[0].requires_grad is True
        ), "Classifier head should be trainable"
    print("  Backbone freezing logic verified.")

    # Unfreeze for training demo
    model.set_backbone_trainable(net, trainable=True)

    # -------------------------------------------------------------------------
    # 5. Engine Verification (Training Loop)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Verifying Training Engine...")

    # Define a minimal stage config for demonstration
    demo_stage_config = {
        "stage_name": "demo_stage",
        "image_size": 224,
        "batch_size": batch_size,
        "epochs": 1,  # Run only 1 epoch
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "patience": 1,
        "checkpoint_name": "demo_best.pth",
        "label_smoothing": 0.0,
    }

    # Run the stage
    # This calls train_one_epoch and validate internally
    trained_model = engine.run_stage(net, train_loader, val_loader, demo_stage_config)

    # Verify Checkpoint Creation
    checkpoint_path = os.path.join(
        config.WORKING_DIR, demo_stage_config["checkpoint_name"]
    )
    if os.path.exists(checkpoint_path):
        print(f"  Checkpoint successfully created at: {checkpoint_path}")
    else:
        raise AssertionError(f"Checkpoint not found at {checkpoint_path}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
