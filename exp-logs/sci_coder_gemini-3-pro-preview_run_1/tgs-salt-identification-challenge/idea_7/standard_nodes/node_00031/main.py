import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.train import train_model
from library.model import HyperResUNet
from library.dataset import get_dataloaders
from library.utils import do_kaggle_metric, unpad_image, rle_encode


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    # Ensure reproducibility
    Config.set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    # The train_model function orchestrates the full training pipeline
    # including saving checkpoints to Config.CHECKPOINT_DIR
    print("\n=== Starting Training Pipeline ===")
    train_model()

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Starting Validation & Failure Analysis ===")

    # Load DataLoaders (Cached)
    # We use load_cached_data=True as requested
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Initialize Models for Snapshot Ensemble
    # We use the best models from Cycle 2 (Robust) and Cycle 3 (Fine-tuned)
    model_c2 = HyperResUNet().to(device)
    model_c3 = HyperResUNet().to(device)

    # Load Checkpoints
    # Fallback to best_model.pth if specific cycle checkpoints aren't found
    # (e.g., if training was cut short or logic changed, though train.py handles this)
    ckpt_c2_path = Config.CYCLE_2_BEST_MODEL
    ckpt_c3_path = Config.CYCLE_3_BEST_MODEL

    if not os.path.exists(ckpt_c2_path):
        ckpt_c2_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_c3_path):
        ckpt_c3_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Loading Model Cycle 2: {ckpt_c2_path}")
    model_c2.load_state_dict(torch.load(ckpt_c2_path, map_location=device))
    model_c2.eval()

    print(f"Loading Model Cycle 3: {ckpt_c3_path}")
    model_c3.load_state_dict(torch.load(ckpt_c3_path, map_location=device))
    model_c3.eval()

    # Validation Inference Loop
    val_preds = []
    val_targets = []

    # Retrieve metadata for failure analysis
    # val_loader is not shuffled, so order matches dataset arrays
    val_depths = val_loader.dataset.depths

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)

            # Test-Time Augmentation (Horizontal Flip)
            images_flip = torch.flip(images, [3])

            # --- Model Cycle 2 ---
            out2 = model_c2(images)
            out2_flip = model_c2(images_flip)
            prob2 = torch.sigmoid(out2)
            prob2_flip = torch.sigmoid(out2_flip)
            # De-augment
            prob2_avg = (prob2 + torch.flip(prob2_flip, [3])) / 2.0

            # --- Model Cycle 3 ---
            out3 = model_c3(images)
            out3_flip = model_c3(images_flip)
            prob3 = torch.sigmoid(out3)
            prob3_flip = torch.sigmoid(out3_flip)
            # De-augment
            prob3_avg = (prob3 + torch.flip(prob3_flip, [3])) / 2.0

            # --- Ensemble ---
            pred_ens = (prob2_avg + prob3_avg) / 2.0

            # Move to CPU and Unpad
            pred_ens = pred_ens.cpu().numpy()
            masks = masks.numpy()

            for b in range(pred_ens.shape[0]):
                # pred_ens is (B, 1, 128, 128)
                p_unpad = unpad_image(
                    pred_ens[b, 0], (Config.ORIG_HEIGHT, Config.ORIG_WIDTH)
                )
                m_unpad = unpad_image(
                    masks[b, 0], (Config.ORIG_HEIGHT, Config.ORIG_WIDTH)
                )

                val_preds.append(p_unpad)
                val_targets.append(m_unpad)

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate Final Metric
    final_metric = do_kaggle_metric(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")

    # Calculate error per image (1 - Average Precision)
    # Re-implement per-image logic from do_kaggle_metric for granularity
    thresholds = np.arange(0.5, 1.0, 0.05)
    errors = []
    coverages = []

    for p, t in zip(val_preds, val_targets):
        # Binarize at 0.5 for IoU calculation base
        p_bin = (p > 0.5).astype(int)
        t_bin = (t > 0.5).astype(int)

        intersection = np.logical_and(p_bin, t_bin).sum()
        union = np.logical_or(p_bin, t_bin).sum()

        iou = intersection / union if union > 0 else 1.0

        # Calculate AP for this image
        matches = iou > thresholds
        ap = matches.mean()
        errors.append(1.0 - ap)

        # Calculate coverage
        coverages.append(t_bin.mean())

    errors = np.array(errors)
    coverages = np.array(coverages)

    # Correlations
    # Note: val_depths is (N,) array
    if len(errors) != len(val_depths):
        print("Warning: Mismatch in validation set size for analysis.")
    else:
        corr_depth, _ = pearsonr(errors, val_depths)
        corr_cov, _ = pearsonr(errors, coverages)

        print(f"Correlation (Error vs Depth): {corr_depth}")
        print(f"Correlation (Error vs Salt Coverage): {corr_cov}")

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    THRESHOLD_SCORE = 0.8156666666666668

    if final_metric > THRESHOLD_SCORE:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        submission_data = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)

                # TTA
                images_flip = torch.flip(images, [3])

                # Model 2
                out2 = model_c2(images)
                out2_flip = model_c2(images_flip)
                prob2 = (
                    torch.sigmoid(out2) + torch.flip(torch.sigmoid(out2_flip), [3])
                ) / 2.0

                # Model 3
                out3 = model_c3(images)
                out3_flip = model_c3(images_flip)
                prob3 = (
                    torch.sigmoid(out3) + torch.flip(torch.sigmoid(out3_flip), [3])
                ) / 2.0

                # Ensemble
                pred_ens = (prob2 + prob3) / 2.0
                pred_ens = pred_ens.cpu().numpy()

                for b in range(len(ids)):
                    img_id = ids[b]

                    # Unpad
                    p_unpad = unpad_image(
                        pred_ens[b, 0], (Config.ORIG_HEIGHT, Config.ORIG_WIDTH)
                    )

                    # Binarize
                    mask_bin = (p_unpad > 0.5).astype(np.uint8)

                    # RLE Encode
                    rle = rle_encode(mask_bin)
                    submission_data.append([img_id, rle])

        # Save
        df_sub = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
