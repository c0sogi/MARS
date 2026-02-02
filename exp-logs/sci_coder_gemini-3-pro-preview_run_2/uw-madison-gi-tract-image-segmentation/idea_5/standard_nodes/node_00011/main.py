import os
import sys
import numpy as np
import pandas as pd
import cv2
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import (
    rle_encode,
    rle_decode,
    dice_coefficient,
    hausdorff_distance_3d,
    keep_largest_connected_component_3d,
)
from library.data_processing import load_and_preprocess_image, set_seed
from library.retrieval_system import AtlasSegmenter


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Starting runfile.py execution...")

    # 2. Initialize and Train (Load Index) Model
    # The retrieval system is "training-free" but requires building/loading the index.
    model = AtlasSegmenter(config=Config)
    try:
        model.fit(load_cached_data=True)
    except Exception as e:
        print(f"Error loading atlas data: {e}")
        return

    # 3. Validation Phase
    print("\n=== Starting Validation ===")
    if not os.path.exists(Config.VAL_METADATA_PATH):
        print("Validation metadata not found. Skipping validation.")
    else:
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)

        # Prepare to collect metrics
        case_scores = []
        slice_errors = []  # For failure analysis

        # Group by Case and Day to handle 3D volumes
        # Each group represents one scan session
        groups = val_df.groupby(["case", "day"])

        for (case_id, day_id), group in groups:
            # Sort by slice index to ensure correct 3D ordering
            group = group.sort_values("slice")

            # Extract unique slices (metadata contains one row per class per slice)
            # We need one entry per image file
            slice_meta = group[
                ["id", "file_path", "slice", "img_height", "img_width"]
            ].drop_duplicates()

            # Determine max slice for relative depth calculation
            max_slice = slice_meta["slice"].max()

            # Containers for 3D volume construction
            # Shape: (Depth, Height, Width, Classes)
            # We use the Config.IMG_SIZE for processing, then resize back if needed,
            # but metrics are usually calculated on the original grid.
            # However, for 3D Hausdorff, consistent grid is better.
            # We will perform metrics on the standardized IMG_SIZE to save time and memory,
            # as resizing masks back to original varying resolutions for every slice is costly.
            # Note: The competition metric technically requires original resolution,
            # but for this baseline estimation, standardized resolution is acceptable.

            vol_preds = []
            vol_true = []

            # Iterate over slices in the case
            for _, row in slice_meta.iterrows():
                # Load Image
                img = load_and_preprocess_image(row["file_path"])

                # Calculate Depth
                depth = row["slice"] / max_slice

                # Predict
                # Returns (H, W, C)
                pred_mask = model.predict_slice(img, depth)
                vol_preds.append(pred_mask)

                # Reconstruct Ground Truth for this slice
                gt_mask = np.zeros(
                    (Config.IMG_SIZE[0], Config.IMG_SIZE[1], len(Config.CLASSES)),
                    dtype=np.uint8,
                )

                # Get RLEs for this slice ID from the group dataframe
                slice_rles = group[group["id"] == row["id"]]

                for c_idx, class_name in enumerate(Config.CLASSES):
                    rle_entry = slice_rles[slice_rles["class"] == class_name][
                        "segmentation"
                    ].values
                    if len(rle_entry) > 0 and pd.notna(rle_entry[0]):
                        # Decode to original size
                        mask_orig = rle_decode(
                            rle_entry[0], (row["img_height"], row["img_width"])
                        )
                        # Resize to standard size (Nearest Neighbor)
                        mask_std = cv2.resize(
                            mask_orig,
                            (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        gt_mask[:, :, c_idx] = mask_std

                vol_true.append(gt_mask)

                # Collect 2D error for failure analysis (using simple Dice)
                # We average over classes for a single scalar error per slice
                slice_dice = dice_coefficient(gt_mask, pred_mask)
                slice_errors.append(
                    {
                        "error": 1.0 - slice_dice,
                        "relative_depth": depth,
                        "slice_index": row["slice"],
                        "img_width": row["img_width"],
                    }
                )

            # Stack to 3D Volume: (Depth, H, W, C)
            vol_preds = np.stack(vol_preds, axis=0)
            vol_true = np.stack(vol_true, axis=0)

            # Apply 3D Post-Processing per class
            for c_idx in range(len(Config.CLASSES)):
                # Extract class volume
                class_vol = vol_preds[:, :, :, c_idx]
                # Clean
                class_vol_clean = keep_largest_connected_component_3d(class_vol)
                vol_preds[:, :, :, c_idx] = class_vol_clean

            # Calculate Metrics per class
            case_dices = []
            case_hds = []

            for c_idx in range(len(Config.CLASSES)):
                y_true = vol_true[:, :, :, c_idx]
                y_pred = vol_preds[:, :, :, c_idx]

                d = dice_coefficient(y_true, y_pred)
                h = hausdorff_distance_3d(y_true, y_pred)

                case_dices.append(d)
                case_hds.append(h)

            # Aggregate for this case
            avg_dice = np.mean(case_dices)
            avg_hd = np.mean(case_hds)

            # Combined Metric: 0.4*Dice + 0.6*(1 - HD)
            # HD is normalized 0-1 (where 1 is bad). So (1-HD) is good.
            score = 0.4 * avg_dice + 0.6 * (1.0 - avg_hd)
            case_scores.append(score)

        # Final Validation Metric
        final_metric = np.mean(case_scores)
        print(f"Final Validation Metric: {final_metric}")

        # 4. Failure Analysis
        print("\n=== Failure Analysis ===")
        err_df = pd.DataFrame(slice_errors)
        if not err_df.empty:
            print("Correlation between Error (1-Dice) and Metadata:")
            features = ["relative_depth", "slice_index", "img_width"]
            for feat in features:
                if feat in err_df.columns:
                    # Drop NaNs just in case
                    valid_data = err_df[[feat, "error"]].dropna()
                    if len(valid_data) > 1:
                        corr, _ = pearsonr(valid_data[feat], valid_data["error"])
                        print(f"  {feat}: {corr:.4f}")
                    else:
                        print(f"  {feat}: Not enough data")
        else:
            print("No error data collected.")

    # 5. Inference on Test Set
    print("\n=== Starting Inference ===")
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print("Test metadata not found. Cannot generate submission.")
        return

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    submission_rows = []

    # Group by Case/Day
    groups = test_df.groupby(["case", "day"])

    for (case_id, day_id), group in groups:
        group = group.sort_values("slice")

        # Unique slices
        slice_meta = group[
            ["id", "file_path", "slice", "img_height", "img_width"]
        ].drop_duplicates()
        max_slice = slice_meta["slice"].max()

        # Store metadata for reconstruction
        slice_infos = []  # List of dicts
        vol_preds = []

        # Predict Loop
        for _, row in slice_meta.iterrows():
            img = load_and_preprocess_image(row["file_path"])
            depth = row["slice"] / max_slice

            pred_mask = model.predict_slice(img, depth)
            vol_preds.append(pred_mask)

            slice_infos.append(
                {
                    "id": row["id"],
                    "orig_h": row["img_height"],
                    "orig_w": row["img_width"],
                }
            )

        # Stack
        vol_preds = np.stack(vol_preds, axis=0)  # (D, H, W, C)

        # Post-Process
        for c_idx in range(len(Config.CLASSES)):
            class_vol = vol_preds[:, :, :, c_idx]
            class_vol_clean = keep_largest_connected_component_3d(class_vol)
            vol_preds[:, :, :, c_idx] = class_vol_clean

        # Encode and Store
        for i, info in enumerate(slice_infos):
            slice_id = info["id"]
            orig_h = info["orig_h"]
            orig_w = info["orig_w"]

            for c_idx, class_name in enumerate(Config.CLASSES):
                # Get the processed mask
                mask_std = vol_preds[i, :, :, c_idx]

                # Resize back to original resolution for submission
                # Use Nearest Neighbor to keep binary
                mask_orig = cv2.resize(
                    mask_std, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

                # Encode
                rle = rle_encode(mask_orig)

                # Add to submission
                submission_rows.append(
                    {"id": slice_id, "class": class_name, "predicted": rle}
                )

    # 6. Save Submission
    print("Saving submission...")
    sub_df = pd.DataFrame(submission_rows)

    # Ensure columns are in correct order
    sub_df = sub_df[["id", "class", "predicted"]]

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Run complete.")


if __name__ == "__main__":
    main()
