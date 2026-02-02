import os
import torch
import pandas as pd
import numpy as np
import sys

# Import provided library modules
from library import config, dataset, model, trainer


def main():
    print("=== Starting Library Usage Demonstration ===")

    # =========================================================================
    # 1. Configuration Optimization
    # =========================================================================
    print("\n[1] Optimizing configuration for rapid demonstration...")

    # Modify config globals to run a fast, minimal version of the pipeline
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 64  # Use a tiny subset of data
    config.BATCH_SIZE = 8  # Small batch size
    config.NUM_EPOCHS_STAGE1 = 1  # 1 Epoch for Stage 1
    config.NUM_EPOCHS_STAGE2 = 1  # 1 Epoch for Stage 2
    config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    config.PRETRAINED = False  # Skip downloading weights for speed

    # Set seed for reproducibility
    config.seed_everything(42)
    print("Configuration updated: DEBUG=True, Epochs=1, Pretrained=False.")

    # =========================================================================
    # 2. Dataset Module Usage & Verification
    # =========================================================================
    print("\n[2] Verifying Dataset module...")

    # Demonstrate Class Weights Calculation
    weights = dataset.calculate_class_weights()
    print(f"Class Weights Shape: {weights.shape}")

    # Validation
    assert isinstance(weights, torch.Tensor), "Weights should be a Tensor"
    assert weights.shape[0] == config.NUM_CLASSES, "Weights dimension mismatch"
    assert not torch.isnan(weights).any(), "Weights contain NaNs"

    # Demonstrate DataLoader Initialization
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

    # Fetch one batch to verify structure
    images, targets = next(iter(train_loader))
    print(f"Batch Shapes -> Images: {images.shape}, Targets: {targets.shape}")

    # Validation
    expected_image_shape = (config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert (
        images.shape == expected_image_shape
    ), f"Expected image shape {expected_image_shape}, got {images.shape}"
    assert targets.shape == (
        config.BATCH_SIZE,
    ), f"Expected target shape ({config.BATCH_SIZE},), got {targets.shape}"
    assert images.dtype == torch.float32, "Images should be float32"
    assert targets.dtype == torch.long, "Targets should be int64"

    # =========================================================================
    # 3. Model Module Usage & Verification
    # =========================================================================
    print("\n[3] Verifying Model module...")

    # Instantiate Model
    net = model.EfficientNetB4Native(num_classes=config.NUM_CLASSES, pretrained=False)
    net = net.to(config.DEVICE)

    # Demonstrate Forward Pass
    images = images.to(config.DEVICE)
    with torch.no_grad():
        outputs = net(images)
    print(f"Model Output Shape: {outputs.shape}")

    # Validation
    assert outputs.shape == (
        config.BATCH_SIZE,
        config.NUM_CLASSES,
    ), "Output shape mismatch"

    # Demonstrate Backbone Freezing
    print("Freezing backbone...")
    net.freeze_backbone()

    # Verify Backbone is frozen (check first parameter of backbone)
    backbone_param = next(net.backbone.parameters())
    assert backbone_param.requires_grad is False, "Backbone parameter should be frozen"

    # Verify Head is trainable
    head_param = next(net.head.parameters())
    assert head_param.requires_grad is True, "Head parameter should be trainable"

    # Demonstrate Unfreezing Blocks
    print("Unfreezing top blocks...")
    net.unfreeze_blocks(n_blocks=1)

    # Verify top block is unfrozen (check last layer of features)
    # EfficientNet features is a Sequential container
    last_block = list(net.backbone.features.children())[-1]
    last_block_param = next(last_block.parameters())
    assert (
        last_block_param.requires_grad is True
    ), "Top block parameter should be unfrozen"

    # =========================================================================
    # 4. Trainer Module Usage (Full Pipeline)
    # =========================================================================
    print("\n[4] Executing Training Pipeline (Integration Test)...")

    # Run the trainer. This uses the modified config settings.
    # It will train for 1 epoch per stage on the small debug dataset.
    trainer.run()

    # =========================================================================
    # 5. Submission Verification
    # =========================================================================
    print("\n[5] Verifying Submission Output...")

    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission File: {sub_df.shape} rows")
    print(sub_df.head(2))

    # Validation
    assert "Id" in sub_df.columns, "Missing 'Id' column"
    assert "Predicted" in sub_df.columns, "Missing 'Predicted' column"
    assert len(sub_df) > 0, "Submission file is empty"
    assert sub_df["Predicted"].dtype in [
        np.int64,
        np.int32,
    ], "Predicted column should be integers"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
