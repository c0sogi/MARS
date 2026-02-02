import os
import sys
import numpy as np
import pandas as pd
import torch
import glob
import cv2
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.train import run_training
from library.inference import predict_and_submit, keep_largest_component
from library.utils import set_seed, rle_decode, compute_dice_score
from library.dataset import get_loaders, MRIDataset, get_transforms, process_metadata
from library.model import AttentionUNet25D


def compute_hausdorff_3d_score(pred_vol, true_vol):
    """
    Computes a normalized 3D Hausdorff score (1 - HD).
    Coordinates are normalized to [0, 1] to provide a bounded score.
    """
    # Get coordinates of non-zero pixels
    p_coords = np.argwhere(pred_vol)
    t_coords = np.argwhere(true_vol)

    # Handle empty cases
    if len(p_coords) == 0 and len(t_coords) == 0:
        return 1.0  # Perfect match (both empty)
    if len(p_coords) == 0 or len(t_coords) == 0:
        return 0.0  # Worst match (one empty, one not)

    # Normalize coordinates to [0, 1]
    # Shape is (Depth, Height, Width) -> (z, y, x)
    d, h, w = pred_vol.shape

    p_norm = p_coords.astype(np.float32)
    p_norm[:, 0] /= d
    p_norm[:, 1] /= h
    p_norm[:, 2] /= w

    t_norm = t_coords.astype(np.float32)
    t_norm[:, 0] /= d
    t_norm[:, 1] /= h
    t_norm[:, 2] /= w

    # Compute directed Hausdorff distances
    d_pt = directed_hausdorff(p_norm, t_norm)[0]
    d_tp = directed_hausdorff(t_norm, p_norm)[0]

    hd = max(d_pt, d_tp)

    # Bounded score: 1 - HD.
    # We clip to 0.0 just in case HD > 1.0 (possible in unit cube diagonal)
    return max(0.0, 1.0 - hd)


def validate_and_analyze():
    print("\n=== Starting Validation and Failure Analysis ===")

    # Load validation metadata
    if not os.path.exists(Config.VAL_METADATA_PATH):
        print("Validation metadata not found.")
        return

    # Load processed metadata (wide format) to match MRIDataset expectation
    val_df, path_lookup = process_metadata(
        Config.VAL_METADATA_PATH, "val_processed.parquet", load_cached_data=True
    )

    # Load Model
    device = torch.device(Config.DEVICE)
    model = AttentionUNet25D()
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Model file not found.")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # We group by case/day to compute 3D metrics
    cases = val_df["case"].unique()
    metrics = []

    # Transforms (Resize + Normalize)
    transforms = get_transforms("valid")

    print(f"Evaluating on {len(cases)} cases...")

    for case_id in cases:
        case_df = val_df[val_df["case"] == case_id].copy()
        days = case_df["day"].unique()

        for day in days:
            day_df = case_df[case_df["day"] == day].sort_values("slice")

            # Prepare volume containers: (Depth, Channels=3, H, W)
            depth = len(day_df)
            pred_vol = np.zeros(
                (depth, 3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8
            )
            true_vol = np.zeros(
                (depth, 3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8
            )

            # Create a temporary dataset for this specific case/day volume
            temp_ds = MRIDataset(
                day_df, path_lookup, transforms=transforms, mode="valid"
            )
            temp_loader = torch.utils.data.DataLoader(
                temp_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
            )

            slice_idx = 0
            with torch.no_grad():
                for images, masks in temp_loader:
                    images = images.to(device)

                    # Predict
                    logits = model(images)
                    probs = torch.sigmoid(logits)
                    preds = (probs > Config.THRESHOLD).float().cpu().numpy()

                    # Ground Truth
                    masks_np = masks.cpu().numpy()

                    batch_size = preds.shape[0]
                    pred_vol[slice_idx : slice_idx + batch_size] = preds
                    true_vol[slice_idx : slice_idx + batch_size] = masks_np

                    slice_idx += batch_size

            # Calculate metrics per class
            classes = ["large_bowel", "small_bowel", "stomach"]

            for c_idx, cls in enumerate(classes):
                p_c = pred_vol[:, c_idx, :, :]
                t_c = true_vol[:, c_idx, :, :]

                # Apply 3D Post-processing to Predictions (Keep Largest Component)
                if Config.KEEP_LARGEST_COMPONENT:
                    p_c = keep_largest_component(p_c)

                # Dice
                dice = compute_dice_score(p_c, t_c)

                # Hausdorff Score
                hd_score = compute_hausdorff_3d_score(p_c, t_c)

                # Combined Metric
                combined = 0.4 * dice + 0.6 * hd_score

                metrics.append(
                    {
                        "case": case_id,
                        "day": day,
                        "class": cls,
                        "dice": dice,
                        "hd_score": hd_score,
                        "combined": combined,
                        # Meta features for failure analysis
                        "num_slices": depth,
                        "pixel_spacing_w": day_df["pixel_spacing_w"].mean(),
                        "img_width": day_df["img_width"].mean(),
                    }
                )

    # Aggregate Metrics
    metrics_df = pd.DataFrame(metrics)
    final_metric = metrics_df["combined"].mean()

    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    print("Correlation of Error (1 - Combined Score) with Metadata:")

    metrics_df["error"] = 1.0 - metrics_df["combined"]
    metrics_df["class_code"] = metrics_df["class"].astype("category").cat.codes

    features = ["num_slices", "pixel_spacing_w", "img_width", "class_code"]
    for feat in features:
        if metrics_df[feat].nunique() > 1:
            corr, _ = pearsonr(metrics_df[feat], metrics_df["error"])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: NaN (No variance)")

    # Show worst performing cases
    print("\nTop 5 Worst Cases:")
    worst = metrics_df.sort_values("combined").head(5)
    print(worst[["case", "day", "class", "combined", "dice", "hd_score"]])


if __name__ == "__main__":
    # 1. Configure for Fast Baseline
    # We reduce epochs to ensure completion within time limits while maintaining reasonable performance.
    print("Configuring fast baseline...")
    Config.NUM_EPOCHS = 3
    Config.BATCH_SIZE = 32

    # 2. Train
    print("Starting Training...")
    # debug=False ensures we train on the full balanced dataset, but for fewer epochs
    run_training(debug=False, load_cached_data=True)

    # 3. Validate & Analyze
    # This step computes the official metric and performs failure analysis
    validate_and_analyze()

    # 4. Inference & Submission
    print("\nStarting Inference...")
    predict_and_submit(load_cached_data=False)
