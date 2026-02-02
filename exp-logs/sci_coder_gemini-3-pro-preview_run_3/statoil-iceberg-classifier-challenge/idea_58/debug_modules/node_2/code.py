import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd

# Import library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.train as train


def main():
    # =========================================================================
    # 1. SETUP & CONFIGURATION OVERRIDES
    # =========================================================================
    print(">>> Setting up demo configuration...")

    # Define a specific directory for this demo execution to avoid overwriting real work
    DEMO_ID = "demo_execution"
    DEMO_DIR = os.path.join(config.WORKING_DIR, DEMO_ID)
    DEMO_CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Ensure directories exist
    os.makedirs(DEMO_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    # Monkey-patch library.train configuration to force a quick run
    # We override the imported constants in the train module directly
    train.CHECKPOINT_DIR = DEMO_CHECKPOINT_DIR
    train.SUBMISSION_DIR = DEMO_SUBMISSION_DIR
    train.NUM_EPOCHS = 1  # Run only 1 epoch
    train.NUM_FOLDS = 2  # Run only 2 folds
    train.IDEA_ID = DEMO_ID

    # Set seed for reproducibility
    utils.set_seed(42)
    logger = utils.setup_logger(os.path.join(DEMO_DIR, "demo.log"))
    logger.info("Configuration patched for fast execution.")

    # =========================================================================
    # 2. VERIFY UTILS
    # =========================================================================
    print("\n>>> Verifying Utils...")
    meter = utils.AverageMeter()
    meter.update(val=10, n=1)
    meter.update(val=20, n=1)
    assert meter.avg == 15.0, f"AverageMeter failed: expected 15.0, got {meter.avg}"
    print("AverageMeter verified.")

    # =========================================================================
    # 3. VERIFY MODEL
    # =========================================================================
    print("\n>>> Verifying Model Architecture...")
    device = config.DEVICE
    net = model.CDICNN().to(device)

    # Create dummy input: Batch Size 4, 3 Channels, 75x75
    dummy_img = torch.randn(4, 3, 75, 75).to(device)
    # Dummy angles: Batch Size 4
    dummy_ang = torch.randn(4).to(device)

    # Forward pass
    output = net(dummy_img, dummy_ang)

    # Check output shape: Should be (4,) as it returns logits for binary class
    assert output.shape == (
        4,
    ), f"Model output shape mismatch: expected (4,), got {output.shape}"
    print("Model forward pass successful. Output shape verified.")

    # =========================================================================
    # 4. VERIFY DATA PIPELINE
    # =========================================================================
    print("\n>>> Verifying Data Pipeline...")
    # Use a small sample size for data loader verification
    debug_size = 32

    # Get dataloaders using the library function
    # Note: This will use cached data if available in ./working/idea_58/ (default cache)
    # or process it. We use the default cache dir to save time.
    train_loader, val_loader, test_loader = data.get_dataloaders(
        debug_sample_size=debug_size
    )

    # Check Train Loader
    images, angles, targets = next(iter(train_loader))
    print(
        f"Train Batch - Images: {images.shape}, Angles: {angles.shape}, Targets: {targets.shape}"
    )

    assert images.shape == (
        config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Train image batch shape incorrect"
    assert angles.shape == (config.BATCH_SIZE,), "Train angle batch shape incorrect"
    assert targets.shape == (config.BATCH_SIZE,), "Train target batch shape incorrect"

    # Check Test Loader (no targets)
    test_images, test_angles = next(iter(test_loader))
    assert test_images.shape[1:] == (3, 75, 75), "Test image dimensions incorrect"
    print("Data Loaders verified.")

    # =========================================================================
    # 5. EXECUTE TRAINING PIPELINE
    # =========================================================================
    print("\n>>> Executing Training Pipeline (Debug Mode)...")
    # Run training with a slightly larger debug sample to ensure at least a few batches
    # We use 64 samples, batch size 32 -> 2 batches per epoch
    try:
        train.run_training(debug_sample_size=64)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise e

    # =========================================================================
    # 6. VALIDATE OUTPUTS
    # =========================================================================
    print("\n>>> Validating Generated Artifacts...")

    # Check Submission File
    submission_path = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(submission_path):
        raise FileNotFoundError(
            f"Submission file was not generated at {submission_path}"
        )

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file found with {len(df_sub)} rows.")

    # Check Checkpoints (We ran 2 folds, so we expect checkpoints for fold 0 and 1)
    for fold in range(2):
        ckpt_path = os.path.join(DEMO_CHECKPOINT_DIR, f"model_best_fold_{fold}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint for fold {fold} not found at {ckpt_path}"
            )
        print(f"Checkpoint for Fold {fold} verified.")

    print("\n>>> Demo Execution Completed Successfully!")


if __name__ == "__main__":
    main()
