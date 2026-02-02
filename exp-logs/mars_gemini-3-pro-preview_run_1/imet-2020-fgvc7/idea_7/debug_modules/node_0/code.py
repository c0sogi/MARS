import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, Mixup, ModelEMA
from library.data_loader import get_loaders
from library.model import ArtworkModel
from library.trainer import Trainer


def run_demo():
    print("==== Starting Artwork Attribute Labeling Demo ====")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset for quick execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.IMG_SIZE = 320  # Ensure consistency with model expectation

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print("    Configuration updated: DEBUG=True, EPOCHS=1, BATCH_SIZE=8")

    # -------------------------------------------------------------------------
    # 2. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loaders...")

    # Force reload to ensure debug subset is used
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Assertions for Train Loader
    assert len(train_loader) > 0, "Train loader is empty!"
    images, labels, ids = next(iter(train_loader))

    print(f"    Train Batch Shape: Images {images.shape}, Labels {labels.shape}")

    # Check shapes: (Batch, Channels, H, W) and (Batch, NumClasses)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Incorrect label shape: {labels.shape}"

    # Check value ranges
    assert (
        images.min() >= -3.0 and images.max() <= 3.0
    ), "Images do not appear normalized correctly."
    assert (
        labels.min() >= 0.0 and labels.max() <= 1.0
    ), "Labels must be binary (0 or 1)."

    print("    Data Loader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = Config.DEVICE
    model = ArtworkModel(pretrained=False).to(
        device
    )  # No need to download weights for shape check
    model.eval()

    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)

    with torch.no_grad():
        logits = model(dummy_input)

    print(f"    Model Output Shape: {logits.shape}")

    assert logits.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(2, Config.NUM_CLASSES)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs!"

    print("    Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Augmentations (Mixup)
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Mixup Augmentation...")

    # Initialize Mixup with 100% probability for testing
    mixup_fn = Mixup(prob=1.0, switch_prob=0.5)

    # Create dummy batch
    x = torch.randn(4, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    y = torch.randint(0, 2, (4, Config.NUM_CLASSES)).float()

    mixed_x, mixed_y = mixup_fn(x, y)

    assert mixed_x.shape == x.shape, "Mixup altered image tensor shape."
    assert mixed_y.shape == y.shape, "Mixup altered label tensor shape."

    print("    Mixup augmentation verification passed.")

    # -------------------------------------------------------------------------
    # 5. Verify Model EMA
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Model EMA...")

    ema = ModelEMA(model, decay=0.999, device=device)

    # Simulate update
    ema.update(model)

    # Check if EMA model parameters exist and match shape
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema_param = dict(ema.module.named_parameters())[name]
            assert (
                ema_param.shape == param.shape
            ), f"EMA parameter shape mismatch for {name}"

    print("    Model EMA verification passed.")

    # -------------------------------------------------------------------------
    # 6. Integration Test: Trainer
    # -------------------------------------------------------------------------
    print("\n[6] Running Trainer Integration Test (Fit & Predict)...")

    # Re-instantiate Trainer (it will re-load loaders with the Config overrides)
    trainer = Trainer()

    # Run Training (1 Epoch as configured)
    print("    Starting training loop...")
    trainer.fit()

    # Run Inference
    print("    Starting inference...")
    trainer.predict()

    # -------------------------------------------------------------------------
    # 7. Verify Submission Output
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Submission File...")

    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)
    print(f"    Submission loaded. Rows: {len(df_sub)}")

    # Check columns
    expected_cols = ["id", "attribute_ids"]
    assert list(df_sub.columns) == expected_cols, f"Invalid columns: {df_sub.columns}"

    # Check content format (attribute_ids should be string or NaN if empty, but usually string)
    # Note: If no attributes predicted, it might be empty string or NaN depending on pandas version/handling
    if len(df_sub) > 0:
        sample_id = df_sub.iloc[0]["id"]
        assert isinstance(sample_id, str), "ID column should be string."

    print("    Submission file verification passed.")
    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
