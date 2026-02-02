import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

# Ensure library path is correct
sys.path.append(os.getcwd())

from library.config import Config
from library.train import run_training
from library.inference import predict_and_submit
from library.dataset import UWMDataset
from library.model import BiSeNet25D
from library.utils import dice_coef


def calculate_normalized_hausdorff_3d(pred_vol, true_vol):
    """
    Calculates 3D Hausdorff distance on volumes with normalized coordinates.
    Uses Distance Transform (EDT) for efficiency instead of point-cloud calculations.
    """
    # Check for empty masks
    p_sum = np.sum(pred_vol)
    t_sum = np.sum(true_vol)

    # If both empty, distance is 0 (perfect match)
    if p_sum == 0 and t_sum == 0:
        return 0.0
    # If one is empty, return max penalty (1.0 in normalized space)
    if p_sum == 0 or t_sum == 0:
        return 1.0

    depth, h, w = pred_vol.shape

    # Spacing for normalization: (1/D, 1/H, 1/W)
    # This treats the volume as a unit cube [0,1]^3
    spacing = (1.0 / depth, 1.0 / h, 1.0 / w)

    # 1. Distance from True to Pred
    # edt calculates distance to nearest background (0).
    # We want distance from True points (1) to nearest Pred point (1).
    # So we compute EDT of (1 - Pred).
    edt_p = ndimage.distance_transform_edt(1 - pred_vol, sampling=spacing)
    # Max distance over all points in True mask
    d_tp = edt_p[true_vol > 0].max()

    # 2. Distance from Pred to True
    edt_t = ndimage.distance_transform_edt(1 - true_vol, sampling=spacing)
    d_pt = edt_t[pred_vol > 0].max()

    return max(d_tp, d_pt)


def main():
    # =========================================================================
    # 1. Configuration Override for Fast Baseline
    # =========================================================================
    print("Configuring Fast Baseline...")
    # Limit epochs and data size to ensure execution within 2 hours
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2000

    # Set seeds for reproducibility
    Config.setup()

    # =========================================================================
    # 2. Training
    # =========================================================================
    print("\n" + "=" * 40)
    print(" STARTING TRAINING ")
    print("=" * 40)
    run_training(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # =========================================================================
    # 3. Validation & Metric Calculation
    # =========================================================================
    print("\n" + "=" * 40)
    print(" STARTING VALIDATION & METRIC CALCULATION ")
    print("=" * 40)

    # Load Val Data
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    if Config.DEBUG:
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Load Model
    device = Config.DEVICE
    model = BiSeNet25D(num_classes=Config.NUM_CLASSES).to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Model file not found! Training might have failed.")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Loader
    val_ds = UWMDataset(val_df, phase="val", load_cached_data=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Accumulate predictions for 3D Metrics
    case_buffer = {}
    slice_stats = []

    print("Running Validation Inference...")
    with torch.no_grad():
        for images, masks, ids in val_loader:
            images = images.to(device)
            # Forward pass
            main_out, _ = model(images)
            preds = torch.sigmoid(main_out).cpu().numpy()
            masks = masks.numpy()

            for i, img_id in enumerate(ids):
                # Parse ID: caseXXX_dayYY_slice_ZZZZ
                parts = img_id.split("_")
                case_day = f"{parts[0]}_{parts[1]}"

                if case_day not in case_buffer:
                    case_buffer[case_day] = {"p": [], "t": [], "ids": []}

                # Binarize predictions
                p_bin = (preds[i] > 0.5).astype(np.uint8)
                t_bin = masks[i].astype(np.uint8)

                case_buffer[case_day]["p"].append(p_bin)
                case_buffer[case_day]["t"].append(t_bin)
                case_buffer[case_day]["ids"].append(img_id)

    # Compute Metrics
    dice_vals = []
    hausdorff_vals = []

    print("Computing 3D Metrics...")
    for case_day, data in case_buffer.items():
        # Stack to (D, C, H, W)
        p_stack = np.stack(data["p"], axis=0)
        t_stack = np.stack(data["t"], axis=0)

        # Calculate metrics per Class
        for c_idx, cls_name in enumerate(Config.CLASS_LABELS):
            p_vol = p_stack[:, c_idx, :, :]
            t_vol = t_stack[:, c_idx, :, :]

            # 1. Dice Coefficient
            d = dice_coef(t_vol, p_vol)
            dice_vals.append(d)

            # 2. 3D Hausdorff Distance
            h = calculate_normalized_hausdorff_3d(p_vol, t_vol)
            hausdorff_vals.append(h)

            # 3. Collect Slice-wise stats for Failure Analysis
            for z in range(p_vol.shape[0]):
                s_dice = dice_coef(t_vol[z], p_vol[z])
                s_area = np.sum(t_vol[z])
                slice_stats.append(
                    {
                        "case": case_day,
                        "class": cls_name,
                        "dice": s_dice,
                        "mask_area": s_area,
                    }
                )

    mean_dice = np.mean(dice_vals)
    mean_hausdorff = np.mean(hausdorff_vals)

    # Metric: 0.4 * Dice + 0.6 * Hausdorff_Score
    # Hausdorff Score is typically 1 - Distance (bounded 0-1)
    h_score = max(0.0, 1.0 - mean_hausdorff)
    final_metric = 0.4 * mean_dice + 0.6 * h_score

    print(f"Final Validation Metric: {final_metric:.10f}")
    print(
        f"Details -> Mean Dice: {mean_dice:.6f}, Mean Hausdorff Dist: {mean_hausdorff:.6f}"
    )

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS ")
    print("=" * 40)

    fa_df = pd.DataFrame(slice_stats)
    if not fa_df.empty:
        # Correlation between Dice and Mask Area
        # Helps identify if small objects are harder to segment
        corr = fa_df["dice"].corr(fa_df["mask_area"])
        print(f"Correlation (Dice vs Mask Area): {corr:.4f}")

        # Class breakdown
        print("Mean Dice by Class:")
        print(fa_df.groupby("class")["dice"].mean())
    else:
        print("No validation data available for failure analysis.")

    # =========================================================================
    # 5. Submission
    # =========================================================================
    print("\n" + "=" * 40)
    print(" GENERATING SUBMISSION ")
    print("=" * 40)

    predict_and_submit(load_cached_data=True, debug=Config.DEBUG)


if __name__ == "__main__":
    main()
