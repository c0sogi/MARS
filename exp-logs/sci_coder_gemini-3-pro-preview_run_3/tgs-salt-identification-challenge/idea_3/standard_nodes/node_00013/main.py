import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add current directory to path
sys.path.append(".")

# Import from provided library files
from library.config import Config
from library.dataset import get_dataloaders, set_seed
from library.train import run_training
from library.utils import calc_map, rle_encode


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Ensure reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Starting execution on {device}...")
    print(
        f"Training for {Config.EPOCHS} epochs (Lovasz start: {Config.LOVASZ_EPOCH_START})"
    )

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    # run_training handles the training loop, model saving, and threshold optimization
    model, best_threshold = run_training()

    # ---------------------------------------------------------
    # 3. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\nStarting Validation and Failure Analysis...")

    # Load dataloaders (reusing cached data)
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    model.eval()
    val_preds = []
    val_targets = []
    val_scores = []  # Store per-image mAP for correlation analysis

    # Validation Inference Loop with TTA
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward Pass (Original)
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Test-Time Augmentation (Horizontal Flip)
            if Config.USE_TTA:
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = torch.sigmoid(logits_flip)
                # Flip predictions back
                probs_flip = torch.flip(probs_flip, dims=[3])
                # Average predictions
                probs = (probs + probs_flip) / 2.0

            # Store for global metric calculation
            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            val_preds.append(probs_np)
            val_targets.append(masks_np)

            # --- Per-Image Metric Calculation for Failure Analysis ---
            batch_size = probs_np.shape[0]

            # Binarize
            preds_bin = (
                (probs_np > best_threshold).astype(np.uint8).reshape(batch_size, -1)
            )
            targets_bin = (masks_np > 0.5).astype(np.uint8).reshape(batch_size, -1)

            # IoU Calculation
            intersection = (preds_bin & targets_bin).sum(axis=1)
            union = (preds_bin | targets_bin).sum(axis=1)

            iou = np.ones(batch_size)
            mask_union = union > 0
            iou[mask_union] = intersection[mask_union] / union[mask_union]

            # mAP over thresholds 0.5 to 0.95
            thresholds = np.arange(0.5, 1.0, 0.05)
            # matches: (Batch, n_thresholds)
            matches = iou[:, None] > thresholds[None, :]
            mean_ap = matches.mean(axis=1)
            val_scores.extend(mean_ap)

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Final Validation Metric (Global Mean)
    # This uses the calc_map utility which computes mean over the batch
    final_metric = calc_map(val_preds, val_targets, threshold=best_threshold)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # --- Failure Analysis ---
    # Load metadata to get features for correlation
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Calculate Error (1 - mAP)
    errors = 1.0 - np.array(val_scores)

    # Create analysis dataframe
    # val_loader.dataset.ids matches the iteration order (shuffle=False)
    analysis_df = pd.DataFrame(
        {
            "id": val_loader.dataset.ids,
            "error": errors,
            "depth": val_loader.dataset.depths,
        }
    )

    # Merge with metadata to get 'coverage'
    analysis_df = analysis_df.merge(val_meta[["id", "coverage"]], on="id", how="left")

    # Calculate correlations
    corr_depth = analysis_df["error"].corr(analysis_df["depth"])
    corr_cov = analysis_df["error"].corr(analysis_df["coverage"])

    print("-" * 40)
    print("Failure Analysis:")
    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Coverage): {corr_cov:.4f}")
    print("-" * 40)

    # ---------------------------------------------------------
    # 4. Submission Generation
    # ---------------------------------------------------------
    if final_metric > 0.806:
        print("Metric threshold met (> 0.806). Generating submission...")

        test_ids_list = []
        test_rles = []

        # Calculate cropping indices to revert padding
        # Pad size was 128, Original size is 101
        pad_total = Config.IMAGE_SIZE - Config.ORIG_IMAGE_SIZE
        pad_top = pad_total // 2
        pad_left = pad_total // 2
        h_orig = Config.ORIG_IMAGE_SIZE
        w_orig = Config.ORIG_IMAGE_SIZE

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)

                # Forward
                logits = model(images)
                probs = torch.sigmoid(logits)

                # TTA
                if Config.USE_TTA:
                    images_flip = torch.flip(images, dims=[3])
                    logits_flip = model(images_flip)
                    probs_flip = torch.sigmoid(logits_flip)
                    probs_flip = torch.flip(probs_flip, dims=[3])
                    probs = (probs + probs_flip) / 2.0

                probs_np = probs.cpu().numpy()  # (B, 1, 128, 128)

                for i in range(len(ids)):
                    # Crop back to 101x101
                    mask_prob = probs_np[i, 0, :, :]
                    mask_prob_cropped = mask_prob[
                        pad_top : pad_top + h_orig, pad_left : pad_left + w_orig
                    ]

                    # Binarize using optimized threshold
                    mask_bin = (mask_prob_cropped > best_threshold).astype(np.uint8)

                    # Run-Length Encoding
                    rle = rle_encode(mask_bin)

                    test_ids_list.append(ids[i])
                    test_rles.append(rle)

        # Save Submission
        sub_df = pd.DataFrame({"id": test_ids_list, "rle_mask": test_rles})
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {final_metric:.4f} is not greater than 0.806. Submission skipped."
        )


if __name__ == "__main__":
    main()
