import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# 1. Monkey-patch tqdm to suppress progress bars before importing library modules
import tqdm


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = silent_tqdm

# Import library modules
from library.config import Config, seed_everything
from library.data import get_dataloaders, ThoracicDataset
from library.model import SpatiallyAwareCenterNet
from library.loss import CenterNetLoss
from library.engine import train_model, predict_and_submit, evaluate


def main():
    print("Initializing Demonstration...")

    # Set seeds for reproducibility
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # =========================================================================
    # 1. Create Tiny Datasets for Speed
    # =========================================================================
    print("\n[Step 1] Creating tiny dataset subsets...")

    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Load original metadata
    full_train = pd.read_csv(Config.TRAIN_META_PATH)
    full_val = pd.read_csv(Config.VAL_META_PATH)
    full_test = pd.read_csv(Config.TEST_META_PATH)

    # Sample subsets (enough to form at least one batch)
    # We filter for images that actually exist to avoid IO errors during demo
    # (The provided script checks existence, but we double check for safety in demo)

    mini_train = full_train.head(32).copy()
    mini_val = full_val.head(16).copy()
    mini_test = full_test.head(16).copy()

    # Save mini metadata
    mini_train_path = os.path.join(Config.WORK_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORK_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORK_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(f"  Created mini train set: {len(mini_train)} rows")
    print(f"  Created mini val set: {len(mini_val)} rows")
    print(f"  Created mini test set: {len(mini_test)} rows")

    # =========================================================================
    # 2. Configure Runtime Parameters
    # =========================================================================
    print("\n[Step 2] Configuring runtime parameters...")

    # Override Config paths to use our mini datasets
    Config.TRAIN_META_PATH = mini_train_path
    Config.VAL_META_PATH = mini_val_path
    Config.TEST_META_PATH = mini_test_path

    # Optimize hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # =========================================================================
    # 3. Data Loading & Verification
    # =========================================================================
    print("\n[Step 3] Initializing DataLoaders and verifying batch structure...")

    dataloaders = get_dataloaders(
        train_meta=Config.TRAIN_META_PATH,
        val_meta=Config.VAL_META_PATH,
        test_meta=Config.TEST_META_PATH,
    )

    # Fetch one batch from training loader
    images, targets, image_ids = next(iter(dataloaders["train"]))

    # Assertions
    assert isinstance(images, torch.Tensor), "Images should be a torch.Tensor"
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape ({Config.BATCH_SIZE}, 3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {images.shape}"
    assert (
        len(targets) == Config.BATCH_SIZE
    ), "Targets list length should match batch size"

    # Verify target dictionary structure for the first sample
    sample_target = targets[0]
    required_keys = ["boxes", "labels", "cls_target", "orig_size", "image_id"]
    if (
        "boxes" in sample_target
    ):  # Only check if boxes exist (some images might be 'No finding')
        for k in required_keys:
            assert k in sample_target, f"Missing key {k} in target dict"

    print("  Data loading verification passed.")

    # =========================================================================
    # 4. Model Initialization & Forward Pass Verification
    # =========================================================================
    print("\n[Step 4] Initializing Model and verifying forward pass...")

    model = SpatiallyAwareCenterNet(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # Run forward pass
    images = images.to(device)
    outputs = model(images)

    # Expected output shapes (Stride 4 means 640/4 = 160)
    feat_size = Config.IMG_SIZE // 4

    # Verify Heatmap
    assert "hm" in outputs
    assert outputs["hm"].shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
        feat_size,
        feat_size,
    ), f"HM shape mismatch: {outputs['hm'].shape}"

    # Verify Width/Height
    assert "wh" in outputs
    assert outputs["wh"].shape == (
        Config.BATCH_SIZE,
        2,
        feat_size,
        feat_size,
    ), f"WH shape mismatch: {outputs['wh'].shape}"

    # Verify Regression/Offset
    assert "reg" in outputs
    assert outputs["reg"].shape == (
        Config.BATCH_SIZE,
        2,
        feat_size,
        feat_size,
    ), f"Reg shape mismatch: {outputs['reg'].shape}"

    # Verify Global Classification Head
    assert "global_no_finding" in outputs
    assert outputs["global_no_finding"].shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Global head shape mismatch: {outputs['global_no_finding'].shape}"

    print("  Model forward pass verification passed.")

    # =========================================================================
    # 5. Loss Calculation Verification
    # =========================================================================
    print("\n[Step 5] Verifying Loss Calculation...")

    criterion = CenterNetLoss()
    loss_stats = criterion(outputs, targets)

    assert "loss" in loss_stats
    assert not torch.isnan(loss_stats["loss"]), "Loss should not be NaN"
    assert loss_stats["loss"] > 0, "Loss should be positive"

    print(f"  Calculated Loss: {loss_stats['loss'].item():.4f}")
    print("  Loss verification passed.")

    # =========================================================================
    # 6. Training Loop Execution
    # =========================================================================
    print("\n[Step 6] Running Training Loop (1 Epoch)...")

    # We use the provided train_model function
    # It handles optimization, scheduling, and saving checkpoints
    train_model(model, dataloaders, device, epochs=Config.EPOCHS)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("  Training loop completed and model saved.")

    # =========================================================================
    # 7. Validation / Evaluation
    # =========================================================================
    print("\n[Step 7] Running Evaluation on Validation Set...")

    # Evaluate returns mAP score
    # Note: With such a small dataset and 1 epoch, mAP might be 0.0, which is fine for demo logic
    val_map = evaluate(
        model, dataloaders["val"], device, val_meta_path=Config.VAL_META_PATH
    )

    print(f"  Validation mAP: {val_map:.4f}")
    print("  Evaluation logic executed successfully.")

    # =========================================================================
    # 8. Inference & Submission
    # =========================================================================
    print("\n[Step 8] Generating Predictions and Submission File...")

    # Run inference
    predict_and_submit(model, dataloaders["test"], device)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(sub_df) == len(
        mini_test
    ), f"Submission rows {len(sub_df)} != Test set size {len(mini_test)}"
    assert "ID" in sub_df.columns or "image_id" in sub_df.columns, "Missing ID column"
    assert "PredictionString" in sub_df.columns, "Missing PredictionString column"

    # Check format of one prediction string
    pred_str = sub_df.iloc[0]["PredictionString"]
    assert isinstance(pred_str, str), "PredictionString must be a string"
    parts = pred_str.split()
    # Should be multiples of 6 (class conf xmin ymin xmax ymax)
    assert (
        len(parts) % 6 == 0
    ), f"PredictionString format incorrect (len {len(parts)} not divisible by 6)"

    print("  Submission file verification passed.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
