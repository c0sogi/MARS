import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config
from library.utils import set_seed, ModelEMA, Mixup
from library.dataset import get_dataloaders
from library.models import get_model
from library.train import run_training
from library.inference import run_inference


def main():
    print("=== Starting Library Demonstration and Verification ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Patching for Fast Demo
    # -------------------------------------------------------------------------
    print("[1/5] Configuring environment for rapid demonstration...")

    # Enable Debug mode to use a small subset of data (100 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100

    # Reduce training complexity
    Config.EPOCHS = 1
    Config.N_FOLDS = 2  # Run 2 folds to verify CV logic
    Config.BATCH_SIZE = 8
    Config.MODEL_ARCHS = ["convnext_tiny"]  # Use single small model

    # Redirect working directories to a separate demo folder
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure clean slate for demo
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration patched. Output directory: ./working/demo_execution")

    # -------------------------------------------------------------------------
    # 2. Verify Data Loading and Transforms
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying Data Loading and Transforms...")

    # Initialize dataloaders (force no cache to test loading logic)
    train_loader, val_loader = get_dataloaders(load_cached_data=False)

    # Fetch a single batch
    images, labels = next(iter(train_loader))

    print(f"  Batch Image Shape: {images.shape}")
    print(f"  Batch Label Shape: {labels.shape}")

    # Assertions
    expected_shape = (Config.BATCH_SIZE, 3, Config.CROP_SIZE, Config.CROP_SIZE)
    assert (
        images.shape == expected_shape
    ), f"Expected image shape {expected_shape}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Expected label shape {(Config.BATCH_SIZE,)}, got {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32 tensor"
    print("Data Loading verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Verify Model and Utilities
    # -------------------------------------------------------------------------
    print("\n[3/5] Verifying Model Architecture and Utilities...")

    # A. Model Construction
    model = get_model("convnext_tiny", pretrained=False, num_classes=1)
    output = model(images)
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    print("  Model forward pass successful.")

    # B. Mixup
    mixup_fn = Mixup(alpha=1.0)
    mixed_x, y_a, y_b, lam = mixup_fn(images, labels)
    assert mixed_x.shape == images.shape, "Mixup output shape mismatch"
    assert 0.0 <= lam <= 1.0, "Mixup lambda out of range"
    print("  Mixup augmentation verified.")

    # C. ModelEMA
    ema = ModelEMA(model, decay=0.5)
    # Capture initial state
    param_key = list(model.state_dict().keys())[0]
    initial_ema_weight = ema.module.state_dict()[param_key].clone()

    # Modify base model
    with torch.no_grad():
        list(model.parameters())[0].add_(1.0)

    # Update EMA
    ema.update(model)
    new_ema_weight = ema.module.state_dict()[param_key]

    # Verify EMA updated but is distinct from base model
    assert not torch.equal(
        initial_ema_weight, new_ema_weight
    ), "EMA weights did not update"
    print("  ModelEMA update logic verified.")

    # -------------------------------------------------------------------------
    # 4. Execute Training Pipeline
    # -------------------------------------------------------------------------
    print("\n[4/5] Executing Training Pipeline (Simulated)...")
    # This runs the full training loop with our patched config (1 epoch, 2 folds)
    run_training()

    # Verify Checkpoints
    fold_0_ckpt = os.path.join(
        Config.CHECKPOINT_DIR, "best_model_convnext_tiny_fold_0.pth"
    )
    fold_1_ckpt = os.path.join(
        Config.CHECKPOINT_DIR, "best_model_convnext_tiny_fold_1.pth"
    )

    assert os.path.exists(fold_0_ckpt), f"Fold 0 checkpoint missing: {fold_0_ckpt}"
    assert os.path.exists(fold_1_ckpt), f"Fold 1 checkpoint missing: {fold_1_ckpt}"
    print("Training complete. Checkpoints verified.")

    # -------------------------------------------------------------------------
    # 5. Execute Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[5/5] Executing Inference Pipeline...")
    # This loads the checkpoints generated above and creates a submission
    run_inference(load_cached_data=False)

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission loaded. Rows: {len(df_sub)}")

    # Check columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "label" in df_sub.columns, "Submission missing 'label' column"

    # Check row count (should match debug sample size for test set)
    # Note: run_inference uses get_test_dataloader which respects Config.DEBUG
    # The test set sample size is min(len(test_df), Config.DEBUG_SAMPLE_SIZE)
    # We set DEBUG_SAMPLE_SIZE = 100.
    assert len(df_sub) == 100, f"Expected 100 predictions, got {len(df_sub)}"

    print("Inference complete. Submission verified.")
    print("\n=== All Tasks Completed Successfully ===")


if __name__ == "__main__":
    main()
