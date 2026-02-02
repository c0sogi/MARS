import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import glob
import gc

# Import library components
from library.config import Config
from library.train import train_model
from library.model import EfficientNetFPN
from library.utils import rle_encode, keep_largest_component_3d, rle_decode
from library.data import get_dataloaders, get_test_dataloader


def calculate_hausdorff_3d(pred_vol, gt_vol):
    """
    Calculates 3D Hausdorff distance on normalized unit cube.
    """
    # Get coordinates of non-zero pixels
    z_p, y_p, x_p = np.where(pred_vol > 0)
    z_g, y_g, x_g = np.where(gt_vol > 0)

    # Handle empty masks
    if len(z_p) == 0 and len(z_g) == 0:
        return 0.0
    if len(z_p) == 0 or len(z_g) == 0:
        return 1.0  # Max penalty for total mismatch

    # Normalize coordinates to [0, 1]
    D, H, W = pred_vol.shape
    pts_p = np.stack([z_p / D, y_p / H, x_p / W], axis=1)
    pts_g = np.stack([z_g / D, y_g / H, x_g / W], axis=1)

    # Downsample for speed if necessary
    max_pts = 1000
    if len(pts_p) > max_pts:
        rng = np.random.default_rng(42)
        pts_p = pts_p[rng.choice(len(pts_p), max_pts, replace=False)]
    if len(pts_g) > max_pts:
        rng = np.random.default_rng(42)
        pts_g = pts_g[rng.choice(len(pts_g), max_pts, replace=False)]

    d_pg = directed_hausdorff(pts_p, pts_g)[0]
    d_gp = directed_hausdorff(pts_g, pts_p)[0]

    # Hausdorff distance is max of directed distances
    hd = max(d_pg, d_gp)

    # Clip to 0-1 range just in case, though unit cube limits it to sqrt(3)
    # The metric expects a bounded 0-1 score, usually 1 - HD or similar.
    # Given the prompt: "normalized by image size to create a bounded 0-1 score"
    # We assume the distance itself is the score component.
    # However, since we need to Combine with Dice (higher is better),
    # and Hausdorff (lower is better), the formula 0.4*Dice + 0.6*HD implies
    # HD here is likely a similarity score (1 - distance).
    # We will return the distance, and handle the inversion in the metric combination.

    return hd


def compute_dice_3d(pred_vol, gt_vol):
    intersection = np.sum(pred_vol * gt_vol)
    union = np.sum(pred_vol) + np.sum(gt_vol)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection) / union


def main():
    # 1. Configuration Override for Fast Baseline
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 32
    print(f"Running with EPOCHS={Config.EPOCHS}, BATCH_SIZE={Config.BATCH_SIZE}")

    # 2. Train Model
    print("\n=== Starting Training ===")
    train_model(epochs=Config.EPOCHS, load_cached_data=True)

    # 3. Validation & Metric Calculation
    print("\n=== Starting Validation & Metric Calculation ===")
    device = torch.device(Config.DEVICE)
    model = EfficientNetFPN(
        encoder_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES
    )
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Get validation loader and dataset to access metadata
    _, val_loader = get_dataloaders(load_cached_data=True)
    val_df = val_loader.dataset.df

    # Collect predictions
    all_preds = []
    all_gts = []

    print("Generating validation predictions...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).cpu().numpy().astype(np.uint8)
            masks = masks.cpu().numpy().astype(np.uint8)

            all_preds.append(preds)
            all_gts.append(masks)

    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    # Group by case for 3D metric
    val_df["temp_idx"] = range(len(val_df))
    cases = val_df["case"].unique()

    dice_scores = []
    hd_scores = []
    case_metrics = []

    print("Calculating 3D metrics...")
    for case in tqdm(cases):
        case_df = val_df[val_df["case"] == case]
        indices = case_df["temp_idx"].values

        # Sort by slice index to ensure correct 3D volume construction
        # The dataframe might be shuffled if not careful, but val_loader was shuffle=False
        # However, let's ensure order by slice number
        slice_indices = case_df["slice"].values
        sorted_order = np.argsort(slice_indices)
        indices = indices[sorted_order]

        case_preds = all_preds[indices]  # (D, C, H, W)
        case_gts = all_gts[indices]  # (D, C, H, W)

        # Transpose to (C, D, H, W) for easier processing per class
        case_preds = case_preds.transpose(1, 0, 2, 3)
        case_gts = case_gts.transpose(1, 0, 2, 3)

        c_dices = []
        c_hds = []

        for c in range(Config.NUM_CLASSES):
            pred_vol = case_preds[c]
            gt_vol = case_gts[c]

            d = compute_dice_3d(pred_vol, gt_vol)
            # Calculate distance
            hd_dist = calculate_hausdorff_3d(pred_vol, gt_vol)
            # Convert distance to score: 1 - distance (clamped at 0)
            # Assuming normalized distance roughly in [0, 1] for good matches
            # If distance > 1, score is 0.
            hd_score = max(0.0, 1.0 - hd_dist)

            c_dices.append(d)
            c_hds.append(hd_score)

        avg_dice = np.mean(c_dices)
        avg_hd = np.mean(c_hds)

        dice_scores.append(avg_dice)
        hd_scores.append(avg_hd)

        # Combined metric: 0.4 * Dice + 0.6 * HausdorffScore
        combined = 0.4 * avg_dice + 0.6 * avg_hd

        case_metrics.append(
            {
                "case": case,
                "dice": avg_dice,
                "hd_score": avg_hd,
                "combined": combined,
                "num_slices": len(indices),
            }
        )

    final_dice = np.mean(dice_scores)
    final_hd = np.mean(hd_scores)
    final_metric = 0.4 * final_dice + 0.6 * final_hd

    print(f"Final Validation Metric: {final_metric:.6f}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    metrics_df = pd.DataFrame(case_metrics)

    # Add metadata features for correlation
    # We aggregate slice metadata to case level
    meta_features = (
        val_df.groupby("case")
        .agg({"img_width": "first", "pixel_spacing_w": "mean"})
        .reset_index()
    )

    analysis_df = metrics_df.merge(meta_features, on="case")
    analysis_df["error"] = 1.0 - analysis_df["combined"]

    # Calculate correlations
    correlations = analysis_df[
        ["error", "num_slices", "img_width", "pixel_spacing_w"]
    ].corr()["error"]
    print("Correlation of Error with features:")
    print(correlations)

    # 5. Submission
    print("\n=== Generating Submission ===")
    test_loader = get_test_dataloader()

    # Collect all predictions
    test_preds = []
    test_ids = []

    with torch.no_grad():
        for images, ids in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            # Keep probabilities for now, threshold later
            preds = preds.cpu().numpy()

            test_preds.append(preds)
            test_ids.extend(ids)

    test_preds = np.concatenate(test_preds, axis=0)  # (N, C, H, W)

    # Map IDs to metadata to reconstruct cases
    # Parse IDs: caseXXX_dayYY_slice_ZZZZ
    parsed_ids = []
    for i, id_str in enumerate(test_ids):
        parts = id_str.split("_")
        # case123_day20_slice_0001
        case_str = parts[0]
        day_str = parts[1]
        slice_num = int(parts[3])
        parsed_ids.append(
            {
                "id": id_str,
                "case": case_str,
                "day": day_str,
                "slice": slice_num,
                "idx": i,
            }
        )

    id_df = pd.DataFrame(parsed_ids)

    # Process by case/day (some cases are split by day in the folder structure logic)
    # Actually, unique identifier for a volume is (case, day)
    groups = id_df.groupby(["case", "day"])

    submission_rows = []

    for (case, day), group in tqdm(groups):
        # Sort by slice
        group = group.sort_values("slice")
        indices = group["idx"].values
        ids = group["id"].values

        # Extract volume: (D, C, H, W)
        vol_preds = test_preds[indices]

        # Process each class
        for c_idx, cls_name in enumerate(Config.CLASSES):
            # Extract class volume (D, H, W)
            cls_vol = vol_preds[:, c_idx, :, :]

            # Threshold
            cls_vol_bin = (cls_vol > Config.THRESHOLD).astype(np.uint8)

            # 3D Post-processing
            if Config.USE_3D_CONN_COMP:
                cls_vol_bin = keep_largest_component_3d(cls_vol_bin)

            # Encode each slice
            for i in range(len(ids)):
                rle = rle_encode(cls_vol_bin[i])
                submission_rows.append(
                    {"id": ids[i], "class": cls_name, "predicted": rle}
                )

    submission_df = pd.DataFrame(submission_rows)

    # Ensure correct column order and format
    submission_df = submission_df[["id", "class", "predicted"]]
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
