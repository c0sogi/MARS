import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, calc_map, optimize_thresholds
from library.dataset import SaltDataset
from library.trainer import Trainer


def main():
    # --- 1. Configuration & Setup ---
    # Set epochs to 30 to ensure runtime is well within 2 hours on A100
    # while providing enough steps for the loss schedule (switch at 15).
    config = Config(epochs=30)
    set_seed(config.SEED)

    # --- 2. Data Loading ---
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)

    # Create Datasets
    # load_cached_data=True is default in Dataset, but explicit here for clarity
    train_ds = SaltDataset(train_df, mode="train", config=config, load_cached_data=True)
    val_ds = SaltDataset(val_df, mode="val", config=config, load_cached_data=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 3. Model Initialization ---
    trainer = Trainer(config)

    # --- 4. Training Loop ---
    print(f"Starting training for {config.EPOCHS} epochs...")
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss, loss_name = trainer.train_one_epoch(train_loader, epoch)

        # Validate
        val_loss, val_score = trainer.validate(val_loader)

        # Scheduler Step
        trainer.scheduler.step(val_score)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} [{loss_name}] - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val mAP: {val_score:.6f} - "
            f"Time: {elapsed:.2f}s"
        )

        # Save Best Model
        if val_score > trainer.best_score:
            trainer.best_score = val_score
            torch.save(
                trainer.model.state_dict(), config.get_model_save_path("best_model.pth")
            )
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping Logic
        # Reset patience if we just switched losses (at epoch 15)
        if epoch == config.LOVASZ_SWITCH_EPOCH:
            print("Loss switched to Lovasz-Hinge. Resetting patience.")
            patience_counter = 0

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(
        f"Training complete. Best Val mAP (during training): {trainer.best_score:.6f}"
    )

    # --- 5. Final Evaluation & Threshold Optimization ---
    print("Loading best model for optimization...")
    trainer.model.load_state_dict(
        torch.load(config.get_model_save_path("best_model.pth"))
    )

    print("Gathering validation predictions...")
    _, _, probs, masks = trainer.validate(val_loader, return_probs=True)

    print("Optimizing threshold...")
    best_threshold = optimize_thresholds(masks, probs, verbose=False)
    trainer.best_threshold = best_threshold
    print(f"Optimal Threshold found: {best_threshold:.4f}")

    # Calculate Final Metric with optimal threshold
    final_metric = calc_map(masks, probs, threshold=best_threshold)
    print(f"Final Validation Metric: {final_metric}")

    # --- 6. Failure Analysis ---
    print("\n--- Failure Analysis ---")

    # Calculate per-image IoU to determine error magnitude
    # Binarize predictions
    preds_bin = (probs > best_threshold).astype(np.uint8)
    masks_bin = masks.astype(np.uint8)

    # Flatten spatial dims: (N, H, W) -> (N, H*W)
    preds_flat = preds_bin.reshape(preds_bin.shape[0], -1)
    masks_flat = masks_bin.reshape(masks_bin.shape[0], -1)

    intersection = (preds_flat * masks_flat).sum(axis=1)
    union = preds_flat.sum(axis=1) + masks_flat.sum(axis=1) - intersection

    # IoU calculation (handle division by zero for empty union)
    ious = np.ones(len(union), dtype=np.float32)
    non_empty = union > 0
    ious[non_empty] = intersection[non_empty] / union[non_empty]

    # Error magnitude (1 - IoU)
    error_magnitude = 1.0 - ious

    # Correlate with Metadata Features
    # Ensure alignment: DataLoader(shuffle=False) preserves order of val_df
    if len(val_df) == len(error_magnitude):
        depths = val_df["z"].values
        coverages = val_df["coverage"].values

        # Calculate correlations
        # np.corrcoef returns matrix [[1, r], [r, 1]]
        corr_depth = np.corrcoef(error_magnitude, depths)[0, 1]
        corr_cov = np.corrcoef(error_magnitude, coverages)[0, 1]

        print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
        print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")
    else:
        print("Error: Validation dataframe length mismatch with predictions.")

    # --- 7. Submission Generation ---
    if final_metric > 0.806:
        print(f"\nMetric {final_metric} > 0.806. Generating submission...")
        trainer.generate_submission()
    else:
        print(f"\nMetric {final_metric} <= 0.806. Skipping submission.")


if __name__ == "__main__":
    main()
