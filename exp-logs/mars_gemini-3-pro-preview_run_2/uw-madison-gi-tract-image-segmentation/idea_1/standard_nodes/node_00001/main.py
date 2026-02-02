import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from library.config import Config
from library.utils import setup_system, hausdorff_3d, dice_coef, rle_decode
from library.trainer import Trainer
from library.inference import generate_submission
from library.dataset import UWDataset
from library.model import MobileNetV2UNet


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast baseline execution
    Config.EPOCHS = 5
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5000  # Limit samples for speed
    Config.BATCH_SIZE = 32

    # Setup system (seeds, directories)
    setup_system()
    print(f"System initialized. Device: {Config.DEVICE}")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n=== Starting Training ===")
    trainer = Trainer(debug=Config.DEBUG, load_cached_data=True)
    trainer.fit()

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    print("\n=== Starting Validation & Metric Calculation ===")

    # Load the full validation set (ignoring debug limit for accurate metric)
    val_dataset = UWDataset(mode="val", debug=False, load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the best model
    model = MobileNetV2UNet(pretrained=False)
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)
    model.eval()

    # Collect predictions
    all_preds = []
    print(f"Running inference on {len(val_dataset)} validation images...")

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(Config.DEVICE)
            with autocast(enabled=True):
                logits = model(images)
                probs = torch.sigmoid(logits)
                preds = (probs > Config.PRED_THRESHOLD).cpu().numpy().astype(np.uint8)
                all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)

    # Reconstruct 3D volumes and calculate metrics
    val_df = val_dataset.df
    scans = val_df.groupby(["case", "day"])

    scan_metrics = []

    print(f"Evaluating metrics on {len(scans)} scans...")

    for (case, day), group in scans:
        # Sort by slice to ensure Z-axis consistency
        group = group.sort_values("slice")
        indices = group.index.values

        # Get predictions for this scan: (Depth, C, H, W)
        scan_preds = all_preds[indices]

        # Reconstruct Ground Truth for this scan
        d, c, h, w = scan_preds.shape
        scan_gt = np.zeros((d, c, h, w), dtype=np.uint8)

        for idx, (_, row) in enumerate(group.iterrows()):
            h_orig, w_orig = row["img_height"], row["img_width"]
            for cls_idx, cls_name in enumerate(Config.CLASS_LABELS):
                seg_col = f"seg_{cls_name}"
                if pd.notna(row.get(seg_col)):
                    # Decode RLE
                    mask = rle_decode(row[seg_col], (h_orig, w_orig))
                    # Resize to model output size (256x256) for comparison
                    mask = cv2.resize(
                        mask, Config.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST
                    )
                    scan_gt[idx, cls_idx] = mask

        # Calculate metrics per class
        dice_sum = 0.0
        hd_sum = 0.0

        for cls_idx in range(Config.NUM_CLASSES):
            y_pred_vol = scan_preds[:, cls_idx]
            y_true_vol = scan_gt[:, cls_idx]

            # Dice
            dice_sum += dice_coef(y_true_vol, y_pred_vol)

            # Hausdorff
            # Note: hausdorff_3d returns a distance.
            # We clip it to 1.0 for the score calculation to avoid negative scores if distance > 1.
            hd_dist = hausdorff_3d(y_true_vol, y_pred_vol)
            hd_sum += min(hd_dist, 1.0)

        avg_dice = dice_sum / Config.NUM_CLASSES
        avg_hd = hd_sum / Config.NUM_CLASSES

        # Metric: 0.4 * Dice + 0.6 * (1 - Hausdorff)
        # We invert Hausdorff because it is a distance (lower is better), but we want a score (higher is better).
        score = 0.4 * avg_dice + 0.6 * (1.0 - avg_hd)

        scan_metrics.append(
            {
                "case": case,
                "day": day,
                "score": score,
                "error": 1.0 - score,
                "slice_count": len(group),
            }
        )

    # Compute Final Metric
    final_metric = np.mean([m["score"] for m in scan_metrics])
    print(f"Final Validation Metric: {final_metric:.18f}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    metrics_df = pd.DataFrame(scan_metrics)

    # Aggregate metadata features per scan
    meta_agg = (
        val_df.groupby(["case", "day"])[["pixel_spacing_w", "img_width"]]
        .mean()
        .reset_index()
    )

    # Merge metrics with features
    analysis_df = pd.merge(metrics_df, meta_agg, on=["case", "day"])

    # Calculate correlations
    features = ["slice_count", "pixel_spacing_w", "img_width"]
    correlations = analysis_df[["error"] + features].corr()["error"].drop("error")

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\n=== Generating Submission ===")
    # Generate submission for the test set
    generate_submission(debug=Config.DEBUG, load_cached_data=True)


if __name__ == "__main__":
    main()
