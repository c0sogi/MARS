import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed, dice_coefficient, hausdorff_distance_3d
from library.dataset import process_metadata, UWDataset, get_transforms
from library.model import LinkNet
from library.train import train
from library.inference import run_inference


def main():
    # ---------------------------------------------------------
    # 1. Configuration for Fast Baseline
    # ---------------------------------------------------------
    # Override Config defaults to ensure completion within 2 hours
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 128  # A100 has 40GB VRAM, can handle large batches
    Config.NUM_WORKERS = 12

    # Ensure reproducibility
    set_seed(Config.SEED)

    print("=== Starting Runfile Execution ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    print("\n--- Phase 1: Training ---")
    train()

    # ---------------------------------------------------------
    # 3. Validation & Metric Calculation
    # ---------------------------------------------------------
    print("\n--- Phase 2: Validation & Metric Assessment ---")

    # Load best model
    device = Config.DEVICE
    model = LinkNet().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print(
            "Warning: Model checkpoint not found. Using untrained model for validation."
        )

    model.eval()

    # Load Validation Data
    # process_metadata handles caching
    val_df = process_metadata(
        Config.VAL_METADATA_PATH, mode="val", load_cached_data=True
    )
    val_dataset = UWDataset(val_df, mode="val", transforms=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Dictionary to aggregate slices into volumes: data[case_day] = list of slices
    case_day_volumes = {}

    print("Running inference on validation set...")
    current_idx = 0

    with torch.no_grad():
        for images, masks in val_loader:
            batch_size = images.size(0)
            images = images.to(device)

            # Predict
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > Config.PRED_THRESHOLD).cpu().numpy().astype(np.uint8)
            masks = masks.cpu().numpy().astype(np.uint8)

            # Map back to metadata to group by case_day
            batch_meta = val_df.iloc[current_idx : current_idx + batch_size]

            for i in range(batch_size):
                row = batch_meta.iloc[i]
                case_day = f"{row['case']}_{row['day']}"
                slice_num = row["slice"]

                if case_day not in case_day_volumes:
                    case_day_volumes[case_day] = []

                case_day_volumes[case_day].append(
                    {
                        "slice": slice_num,
                        "pred": preds[i],  # (C, H, W)
                        "gt": masks[i],  # (C, H, W)
                    }
                )

            current_idx += batch_size

    # Compute 3D Metrics
    print("Computing 3D Metrics...")
    dice_scores = []
    hausdorff_scores = []

    # For Failure Analysis
    fa_records = []

    for case_day, slices in case_day_volumes.items():
        # Sort slices by Z index to form proper 3D volume
        slices.sort(key=lambda x: x["slice"])

        # Stack to create volumes: Shape (D, C, H, W) -> Transpose to (C, D, H, W)
        vol_pred = np.stack([s["pred"] for s in slices], axis=1)
        vol_gt = np.stack([s["gt"] for s in slices], axis=1)

        # Iterate over classes (Large Bowel, Small Bowel, Stomach)
        for c_idx, class_name in enumerate(Config.CLASS_LABELS):
            p = vol_pred[c_idx]  # (D, H, W)
            g = vol_gt[c_idx]  # (D, H, W)

            # 3D Dice Calculation
            intersection = np.sum(p * g)
            union = np.sum(p) + np.sum(g)

            # Dice is 0 if both empty
            if union == 0:
                dice = 0.0
            else:
                dice = (2.0 * intersection) / union

            # 3D Hausdorff Calculation
            hd = hausdorff_distance_3d(p, g)

            dice_scores.append(dice)
            hausdorff_scores.append(hd)

            # Collect data for failure analysis
            # Feature: Volume of the object (sum of pixels)
            mask_volume = np.sum(g)
            fa_records.append(
                {
                    "case_day": case_day,
                    "class": class_name,
                    "dice": dice,
                    "error": 1.0 - dice,
                    "mask_volume": mask_volume,
                    "hd": hd,
                }
            )

    # Aggregate Metrics
    mean_dice = np.mean(dice_scores)
    mean_hd = np.mean(hausdorff_scores)

    # Metric Combination: 0.4 * Dice + 0.6 * Hausdorff
    # Note: Hausdorff is a distance (lower is better), but usually competition metrics
    # normalize it to a score (1 - HD) or similar.
    # Given the prompt "normalized... to create a bounded 0-1 score", and typical Dice combinations,
    # we assume the score formulation: Score = 0.4 * Dice + 0.6 * (1 - HD).
    # We clip HD to 1.0 to ensure the score doesn't go negative if the distance is large.
    final_metric = 0.4 * mean_dice + 0.6 * (1.0 - min(1.0, mean_hd))

    print(f"Mean Dice: {mean_dice:.6f}")
    print(f"Mean Hausdorff: {mean_hd:.6f}")
    print(f"Final Validation Metric: {final_metric:.18f}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Phase 3: Failure Analysis ---")
    fa_df = pd.DataFrame(fa_records)

    # Filter for objects that actually exist in GT to analyze sensitivity to size
    # (Empty masks have volume 0 and usually Dice 0 or 1 depending on prediction)
    valid_objects = fa_df[fa_df["mask_volume"] > 0]

    if len(valid_objects) > 0:
        corr_volume = valid_objects["error"].corr(valid_objects["mask_volume"])
        print(
            f"Correlation between Error (1-Dice) and Object Volume: {corr_volume:.4f}"
        )
    else:
        print("No valid objects found for correlation analysis.")

    # ---------------------------------------------------------
    # 5. Submission
    # ---------------------------------------------------------
    print("\n--- Phase 4: Submission Generation ---")
    run_inference(load_cached_data=True)

    print("\n=== Runfile Execution Complete ===")


if __name__ == "__main__":
    main()
