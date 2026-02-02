import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import library components
import library.config as config
from library.dataset import SaltDataset
from library.model import DepthRobustLinkNet
from library.losses import CombinedLoss
from library.train import train_one_epoch, validate, set_seed
from library.utils import do_kaggle_metric, rle_encode, save_checkpoint

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Reduce epochs to ensure execution within time limits while maintaining performance
config.EPOCHS = 20
config.SCHEDULER_T_MAX = config.EPOCHS


def get_crop_indices(target_size=101, padded_size=128):
    """
    Determines the slicing indices to crop the center of the image
    matching Albumentations PadIfNeeded(position='center') behavior.
    """
    delta = padded_size - target_size
    pad_top = delta // 2
    pad_bottom = delta - pad_top
    pad_left = delta // 2
    pad_right = delta - pad_left
    return pad_top, padded_size - pad_bottom, pad_left, padded_size - pad_right


def run_failure_analysis(model, val_loader, device, threshold):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between Error (1-IoU) and metadata features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    ious = []
    depths = []
    coverages = []

    # We need to access the original metadata to get salt coverage
    # The dataset loader provides (image, mask, depth, id)
    # We can link back to metadata using the ID or just calculate coverage from the mask tensor

    with torch.no_grad():
        for images, masks, batch_depths, _ in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            batch_depths = batch_depths.to(device)

            logits = model(images, batch_depths)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # Calculate IoU per image in batch
            # shape: (B, 1, H, W)
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = (preds + masks).clamp(0, 1).sum(dim=(1, 2, 3))

            # Handle division by zero (empty union means IoU=1 if both empty)
            batch_ious = torch.ones_like(intersection)
            non_empty = union > 0
            batch_ious[non_empty] = intersection[non_empty] / union[non_empty]

            ious.extend(batch_ious.cpu().numpy())
            depths.extend(batch_depths.cpu().numpy().flatten())

            # Calculate coverage from ground truth mask
            # mask shape (1, H, W) -> sum / total_pixels
            batch_cov = masks.sum(dim=(1, 2, 3)) / (masks.shape[2] * masks.shape[3])
            coverages.extend(batch_cov.cpu().numpy())

    ious = np.array(ious)
    errors = 1.0 - ious
    depths = np.array(depths)
    coverages = np.array(coverages)

    # Calculate correlations
    # Depth is standardized, but correlation is scale invariant
    corr_depth, _ = pearsonr(errors, depths)
    corr_cov, _ = pearsonr(errors, coverages)

    print(f"Correlation [Error vs Depth]: {corr_depth:.4f}")
    print(f"Correlation [Error vs Salt Coverage]: {corr_cov:.4f}")

    if abs(corr_depth) > 0.2:
        print("-> Significant correlation with depth detected.")
    if abs(corr_cov) > 0.2:
        print("-> Significant correlation with salt coverage detected.")


def main():
    # 1. Setup
    set_seed(config.SEED)
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    device = config.DEVICE
    print(f"Running on {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset = SaltDataset(mode="train", load_cached_data=True)
    val_dataset = SaltDataset(mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = DepthRobustLinkNet(in_channels=1, n_classes=1).to(device)

    criterion = CombinedLoss(bce_weight=0.5, lovasz_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.SCHEDULER_T_MAX, eta_min=1e-6
    )

    # 4. Training Loop
    best_map = 0.0
    best_threshold = 0.5

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(1, config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_map, epoch_thresh = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch}/{config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.4f} | Thresh: {epoch_thresh:.2f}"
        )

        # Save Best
        if val_map > best_map:
            best_map = val_map
            best_threshold = epoch_thresh
            save_checkpoint(model, optimizer, epoch, val_map, config.CHECKPOINT_PATH)

    print("-" * 30)
    print(
        f"Training Complete. Best mAP: {best_map:.6f} at Threshold: {best_threshold:.2f}"
    )

    # 5. Final Validation & Failure Analysis
    print("Loading best model for final verification...")
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Re-calculate metric on full set to be absolutely sure and print required format
    _, final_metric, _ = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    run_failure_analysis(model, val_loader, device, best_threshold)

    # 6. Submission
    SUBMISSION_THRESHOLD = 0.7985

    if final_metric > SUBMISSION_THRESHOLD:
        print("\nMetric condition met. Generating submission...")

        # Load Test Data
        test_dataset = SaltDataset(mode="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        model.eval()
        submission_rows = []

        # Calculate crop indices for 128 -> 101
        y1, y2, x1, x2 = get_crop_indices(config.ORIG_SIZE, config.IMG_SIZE)

        with torch.no_grad():
            for images, _, _, ids in test_loader:
                images = images.to(device)

                # Test-Time Augmentation (TTA): Original + Horizontal Flip
                # 1. Forward Pass Original
                # Inject constant depth (0.0) for robustness
                depths = torch.full(
                    (images.size(0), 1), config.DEPTH_FILL_VALUE, device=device
                )
                logits_orig = model(images, depths)
                probs_orig = torch.sigmoid(logits_orig)

                # 2. Forward Pass Flipped
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip, depths)
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip = torch.flip(probs_flip, dims=[3])

                # Average
                probs_avg = (probs_orig + probs_flip) / 2.0

                # Post-processing
                # Crop back to 101x101
                probs_cropped = probs_avg[:, :, y1:y2, x1:x2]

                # Threshold
                pred_masks = (probs_cropped > best_threshold).float().cpu().numpy()

                # Encode
                for i in range(len(ids)):
                    mask = pred_masks[i, 0, :, :]  # (101, 101)
                    rle = rle_encode(mask)
                    submission_rows.append([ids[i], rle])

        # Save Submission
        sub_df = pd.DataFrame(submission_rows, columns=["id", "rle_mask"])
        sub_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH} with {len(sub_df)} rows.")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
