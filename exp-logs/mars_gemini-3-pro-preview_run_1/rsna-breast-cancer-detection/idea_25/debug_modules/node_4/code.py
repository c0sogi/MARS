import os
import sys

# Cite Lesson 13 & 16: Preempt TensorFlow Memory Allocation
try:
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
except ImportError:
    pass

import torch
import numpy as np
import pandas as pd
import shutil
import gc

# Import provided library modules
from library import config, utils, data, model, train


def cleanup_memory():
    # Cite debug_lesson_18: Purge System Tracebacks to Release Zombie GPU Memory
    if hasattr(sys, "last_traceback"):
        sys.last_traceback = None

    # Cite debug_lesson_7 & 15: Purge Global Variables (including Optimizers/Containers)
    gl = globals()
    keys = list(gl.keys())
    for key in keys:
        if key in [
            "sys",
            "os",
            "torch",
            "gc",
            "shutil",
            "pd",
            "np",
            "config",
            "utils",
            "data",
            "model",
            "train",
            "tf",
        ]:
            continue
        if key.startswith("__"):
            continue
        obj = gl.get(key)
        if isinstance(obj, (torch.nn.Module, torch.Tensor, torch.optim.Optimizer)):
            del gl[key]

    gc.collect()
    torch.cuda.empty_cache()


def main():
    cleanup_memory()
    print("Starting demonstration of Breast Cancer Detection pipeline...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Modify configuration for a quick demonstration run
    print("\n[1] Configuring environment for demo...")

    # Override config parameters for speed
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 2  # Small batch size for demo
    config.DEBUG = True

    # Ensure output directories are clean/ready
    if os.path.exists(config.SUBMISSION_PATH):
        os.remove(config.SUBMISSION_PATH)

    # Set random seeds for reproducibility
    utils.seed_everything(config.SEED)
    print("Configuration updated: Epochs=1, BatchSize=2, Debug=True")

    # =========================================================================
    # 2. Validate Metric Logic
    # =========================================================================
    print("\n[2] Validating Probabilistic F1 Score (pF1)...")

    # Case 1: Perfect prediction
    y_true_perfect = np.array([1, 0, 1, 0])
    y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    pf1_perfect = utils.probabilistic_f1(y_true_perfect, y_pred_perfect)

    # Case 2: Random/Mixed prediction
    y_true_mixed = np.array([1, 0, 1, 0])
    y_pred_mixed = np.array([0.8, 0.2, 0.6, 0.4])
    pf1_mixed = utils.probabilistic_f1(y_true_mixed, y_pred_mixed)

    print(f"pF1 (Perfect): {pf1_perfect:.4f}")
    print(f"pF1 (Mixed):   {pf1_mixed:.4f}")

    # Assertions
    assert np.isclose(pf1_perfect, 1.0), "pF1 should be 1.0 for perfect predictions"
    assert 0.0 <= pf1_mixed <= 1.0, "pF1 should be between 0 and 1"
    print("Metric validation passed.")

    # =========================================================================
    # 3. Data Pipeline Inspection
    # =========================================================================
    print("\n[3] Inspecting Data Pipeline...")

    # Use a tiny subset for inspection
    subset_size = 10
    train_loader, val_loader, test_loader = data.get_dataloaders(
        load_cached_data=False, debug_subset_size=subset_size
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    images = batch["image"]
    images_contra = batch["image_contra"]
    labels = batch["label"]

    print(f"Batch keys: {batch.keys()}")
    print(f"Image Shape: {images.shape}")
    print(f"Contra Image Shape: {images_contra.shape}")
    print(f"Label Shape: {labels.shape}")

    # Assertions for data integrity
    # Shape: (Batch, Channels, Height, Width) -> (2, 3, 768, 768)
    expected_shape = (
        config.BATCH_SIZE,
        config.IN_CHANNELS,
        config.IMAGE_SIZE[0],
        config.IMAGE_SIZE[1],
    )
    assert (
        images.shape == expected_shape
    ), f"Expected image shape {expected_shape}, got {images.shape}"
    assert (
        images_contra.shape == expected_shape
    ), f"Expected contra shape {expected_shape}, got {images_contra.shape}"
    assert labels.shape[0] == config.BATCH_SIZE, "Label batch size mismatch"

    # Verify channels (Channel 1 and 2 should be metadata maps)
    # Channel 1: Age (should be constant per image)
    age_map = images[0, 1, :, :]
    assert torch.min(age_map) == torch.max(
        age_map
    ), "Age map channel should be spatially constant"

    print("Data pipeline inspection passed.")

    # =========================================================================
    # 4. Model Architecture Inspection
    # =========================================================================
    print("\n[4] Inspecting Model Architecture...")

    device = torch.device(config.DEVICE)
    net = model.SiameseEfficientNet().to(device)

    # Move batch to device
    img_dev = images.to(device)
    img_contra_dev = images_contra.to(device)

    # Forward pass
    with torch.no_grad():
        logits = net(img_dev, img_contra_dev)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Expected logits shape ({config.BATCH_SIZE}, 1), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    print("Model forward pass successful.")

    # =========================================================================
    # 5. Full Training Loop Execution
    # =========================================================================
    print("\n[5] Executing Training Loop (Demo Run)...")

    # We use a slightly larger subset for the training run to ensure valid/test batches exist
    # run_training handles the full lifecycle: Train -> Val -> Save Checkpoint -> Generate Submission
    try:
        train.run_training(load_cached_data=False, debug_subset_size=20)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e
    finally:
        cleanup_memory()

    # =========================================================================
    # 6. Submission Verification
    # =========================================================================
    print("\n[6] Verifying Submission Output...")

    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    # Verify columns
    required_cols = ["prediction_id", "cancer"]
    if not all(col in df_sub.columns for col in required_cols):
        raise ValueError(
            f"Submission missing required columns. Found: {df_sub.columns}"
        )

    # Verify values are probabilities
    if not df_sub.empty:
        probs = df_sub["cancer"]
        if probs.min() < 0 or probs.max() > 1:
            raise ValueError(
                "Submission contains values outside [0, 1] probability range."
            )

    print("Submission verification passed.")
    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
