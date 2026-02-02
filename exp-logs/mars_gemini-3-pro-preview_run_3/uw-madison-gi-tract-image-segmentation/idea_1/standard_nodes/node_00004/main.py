import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader
from scipy.spatial.distance import directed_hausdorff

# Import library components
from library.config import Config
from library.utils import set_seed, rle_decode
from library.dataset import (
    process_and_cache_25d_metadata,
    GI_MRI_Dataset,
    get_transforms,
)
from library.model import FPN
from library.train import train_model
from library.inference import run_inference


def compute_metrics_for_volume(pred_vol, true_vol, height, width):
    """
    Computes Dice and Normalized 3D Hausdorff for a single class volume.
    pred_vol, true_vol: (Depth, Height, Width) binary masks (uint8)
    """
    # 1. Dice Coefficient
    # Formula: 2*|X n Y| / (|X| + |Y|)
    # Constraint: 0 if both X and Y are empty.
    intersection = np.sum(pred_vol * true_vol)
    sum_pred = np.sum(pred_vol)
    sum_true = np.sum(true_vol)

    if sum_pred == 0 and sum_true == 0:
        dice = 0.0
    else:
        dice = (2.0 * intersection) / (sum_pred + sum_true)

    # 2. 3D Hausdorff Distance
    # Constraint: Coordinates normalized by image size. Slice depth set to 1.
    # Optimization: Use contours to reduce point count.

    def get_points(vol):
        points = []
        # Iterate over slices
        for z in range(vol.shape[0]):
            slice_mask = vol[z]
            if np.sum(slice_mask) > 0:
                # Find contours
                contours, _ = cv2.findContours(
                    slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in contours:
                    # cnt shape: (N, 1, 2) -> (x, y)
                    # We need (z, y, x) normalized
                    # x normalized = x / width
                    # y normalized = y / height
                    # z = z (slice index)

                    # Extract coords
                    coords = cnt.squeeze(1).astype(np.float32)  # (N, 2) -> (x, y)

                    # Normalize x and y
                    coords[:, 0] /= width
                    coords[:, 1] /= height

                    # Create z column
                    z_col = np.full((coords.shape[0], 1), z, dtype=np.float32)

                    # Combine to (z, y, x) to match standard (D, H, W) logic,
                    # though Hausdorff is symmetric so order of dims only matters for consistency.
                    # Let's use (z, y, x)
                    pts = np.hstack([z_col, coords[:, 1:2], coords[:, 0:1]])
                    points.append(pts)

        if points:
            return np.concatenate(points, axis=0)
        return np.empty((0, 3), dtype=np.float32)

    pred_points = get_points(pred_vol)
    true_points = get_points(true_vol)

    if len(pred_points) == 0 and len(true_points) == 0:
        # Both empty -> Distance is 0
        hd = 0.0
    elif len(pred_points) == 0 or len(true_points) == 0:
        # One empty -> Max penalty. Since normalized 0-1 (mostly), 1.0 is a reasonable cap
        # but Z is unbounded. However, standard metric usually caps at 1.0 for the score calculation.
        hd = 1.0
    else:
        d1 = directed_hausdorff(pred_points, true_points)[0]
        d2 = directed_hausdorff(true_points, pred_points)[0]
        hd = max(d1, d2)

    # Normalize HD score to 0-1 range for the metric formula: 0.6 * (1 - HD)
    # We clip HD at 1.0.
    hd_score = 1.0 - min(1.0, hd)

    return dice, hd_score


def validate_and_analyze():
    print("\n=== Starting Validation and Failure Analysis ===")

    # 1. Load Metadata and Dataset
    # process_and_cache_25d_metadata returns df sorted by case, day, slice
    _, val_df, _ = process_and_cache_25d_metadata(load_cached_data=True)

    # Group by case+day to identify volumes
    val_df["group_id"] = val_df["case"] + "_" + val_df["day"]
    unique_groups = val_df["group_id"].unique()

    print(f"Validation set: {len(val_df)} slices, {len(unique_groups)} volumes.")

    # 2. Load Model
    device = torch.device(Config.DEVICE)
    model = FPN(
        backbone_name=Config.BACKBONE, pretrained=False, num_classes=Config.NUM_CLASSES
    )
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print("Error: Best model not found. Skipping validation.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Inference & Metric Loop
    # We iterate group by group to construct volumes

    val_dataset = GI_MRI_Dataset(val_df, transforms=get_transforms("val"), mode="val")
    # We use a loader but we need to map outputs back to the dataframe structure
    # Since val_df is sorted and Dataset respects index, we can iterate sequentially.

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Store all predictions in a list first (memory permitting for validation set ~7k images)
    # 7k * 320 * 320 * 3 bytes is ~2GB. Feasible on 220GB RAM.
    all_preds = []

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            preds = model(images)  # (B, 3, 320, 320)
            preds = torch.sigmoid(preds).cpu().numpy()
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 3, 320, 320)

    # 4. Compute Metrics per Volume
    metrics_data = []

    # Map slice indices to groups
    group_indices = val_df.groupby("group_id").indices  # dict: group_id -> index array

    total_dice = 0.0
    total_hd_score = 0.0
    count = 0

    print("Computing 3D metrics...")

    for group_id, indices in group_indices.items():
        # Get subset of dataframe
        group_df = val_df.iloc[indices]

        # Get predictions for this group
        # Shape: (D, 3, 320, 320)
        group_preds_raw = all_preds[indices]

        # Get original dimensions (all slices in a scan have same dims)
        orig_h = group_df.iloc[0]["height"]
        orig_w = group_df.iloc[0]["width"]

        # Prepare volumes
        # We need to resize predictions to original size and threshold
        # Also decode ground truth

        depth = len(indices)

        # Containers for volumes (D, H, W) per class
        vol_preds = {
            c: np.zeros((depth, orig_h, orig_w), dtype=np.uint8) for c in Config.CLASSES
        }
        vol_true = {
            c: np.zeros((depth, orig_h, orig_w), dtype=np.uint8) for c in Config.CLASSES
        }

        for i, (idx, row) in enumerate(group_df.iterrows()):
            # Resize prediction
            # group_preds_raw[i] is (3, 320, 320)
            # Transpose to (320, 320, 3) for cv2
            p = np.transpose(group_preds_raw[i], (1, 2, 0))
            p_resized = cv2.resize(p, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            p_mask = (p_resized > 0.5).astype(np.uint8)  # (H, W, 3)

            for c_idx, cls in enumerate(Config.CLASSES):
                vol_preds[cls][i] = p_mask[:, :, c_idx]

                # Decode Ground Truth
                rle = row[cls]
                vol_true[cls][i] = rle_decode(rle, (orig_h, orig_w))

        # Compute metrics for each class
        group_dice = 0
        group_hd = 0

        # Metadata for failure analysis
        total_mask_pixels = sum([np.sum(vol_true[c]) for c in Config.CLASSES])

        for cls in Config.CLASSES:
            d, h_score = compute_metrics_for_volume(
                vol_preds[cls], vol_true[cls], orig_h, orig_w
            )

            # Weighted combination per class?
            # Task says: "The two metrics are combined, with a weight of 0.4 for the Dice metric and 0.6 for the Hausdorff distance."
            # Usually this is done per sample then averaged, or averaged then combined.
            # Standard is per-sample combination.

            score = 0.4 * d + 0.6 * h_score

            metrics_data.append(
                {
                    "case": group_id,
                    "class": cls,
                    "dice": d,
                    "hd_score": h_score,
                    "score": score,
                    "mask_area": np.sum(vol_true[cls]),
                    "depth": depth,
                }
            )

            group_dice += d
            group_hd += h_score

        # Average over 3 classes for this volume
        # (Used for running average if needed, but we calculate global mean from list)

    # 5. Final Metric Calculation
    metrics_df = pd.DataFrame(metrics_data)

    # Mean over all rows (samples * classes)
    final_dice = metrics_df["dice"].mean()
    final_hd = metrics_df["hd_score"].mean()
    final_score = 0.4 * final_dice + 0.6 * final_hd

    print(f"Final Validation Metric: {final_score:.8f}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate correlation between Error (1 - Score) and features
    metrics_df["error"] = 1.0 - metrics_df["score"]

    # Correlation with Mask Area
    corr_area = metrics_df["error"].corr(metrics_df["mask_area"])
    print(f"Correlation between Error and Organ Mask Area: {corr_area:.4f}")

    # Correlation with Depth (Slice Count)
    corr_depth = metrics_df["error"].corr(metrics_df["depth"])
    print(f"Correlation between Error and Scan Depth: {corr_depth:.4f}")

    # Identify worst classes
    print("\nMean Score by Class:")
    print(metrics_df.groupby("class")["score"].mean())

    return final_score


def main():
    # 1. Configure for Fast Baseline
    # Increase epochs to allow convergence with new normalization
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 32

    # Setup directories
    Config.setup(training=True)
    set_seed(Config.SEED)

    print("=== Configuration ===")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # 2. Train Model
    print("\n=== Starting Training ===")
    # debug=False ensures we use the full dataset (or as configured) but with our custom EPOCHS
    best_dice = train_model(debug=False)
    print(f"Training finished. Best Val Dice (approx): {best_dice:.4f}")

    # 3. Validate and Analyze
    # We capture the output to check the metric
    # Note: validate_and_analyze prints the metric, but we can't easily capture stdout here without redirect.
    # We will rely on the printed output for the user, but for submission logic we need to be careful.
    # Since we can't modify validate_and_analyze return value easily without changing signature in runfile (which is allowed),
    # let's just run inference unconditionally if it finishes without error, OR we can parse the logs?
    # Actually, the requirement says "Generate predictions... If and only if the final validation metric is higher than 0.00463659".
    # I will modify validate_and_analyze to return the score.

    final_score = validate_and_analyze()

    # 4. Inference and Submission
    if final_score is not None and final_score > 0.00463659:
        print(f"\n=== Generating Submission (Score {final_score:.5f} > Baseline) ===")
        run_inference(debug=False)
    else:
        print(
            f"\n=== Skipping Submission (Score {final_score} <= Baseline 0.00463659) ==="
        )


if __name__ == "__main__":
    main()
