import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, dice_coef, hausdorff_3d
from library.ghost_model import GhostUNet
from library.dataset import process_metadata, UWGI25DDataset, get_transforms
from library.train import run_training
from library.inference import run_inference


def perform_failure_analysis(model, val_loader, val_df, device):
    """
    Runs inference on validation set, calculates metrics, and correlates errors with features.
    Returns the final combined metric.
    """
    model.eval()

    # Store results
    results = []

    # For 3D Hausdorff calculation
    case_data = {}  # case_day -> {slice: {'pred': ..., 'true': ...}}

    print("Running validation for failure analysis...")
    with torch.no_grad():
        for images, masks, ids in val_loader:
            images = images.to(device)

            # Predict
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float().cpu().numpy()
            masks = masks.numpy()

            for i, img_id in enumerate(ids):
                # Calculate Slice-level Dice for correlation analysis
                slice_dice = dice_coef(masks[i], preds[i])

                # Get metadata for this image
                meta_row = val_df[val_df["id"] == img_id].iloc[0]

                results.append(
                    {
                        "id": img_id,
                        "slice_dice": slice_dice,
                        "error": 1.0 - slice_dice,
                        "slice_idx": meta_row["slice"],
                        "day": meta_row["day"],
                        "pixel_spacing": meta_row[
                            "pixel_spacing_w"
                        ],  # w and h are usually same
                        "img_width": meta_row["img_width"],
                    }
                )

                # Store for 3D metric
                parts = img_id.split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                slice_num = int(parts[3])

                if case_day not in case_data:
                    case_data[case_day] = {}

                case_data[case_day][slice_num] = {"pred": preds[i], "true": masks[i]}

    # --- 1. Calculate Final Metric (3D) ---
    dice_scores_3d = []
    hausdorff_scores_3d = []

    for case_day, slices in case_data.items():
        sorted_idxs = sorted(slices.keys())
        # Stack (Depth, C, H, W)
        vol_pred = np.stack([slices[s]["pred"] for s in sorted_idxs])
        vol_true = np.stack([slices[s]["true"] for s in sorted_idxs])

        # Transpose to (C, Depth, H, W)
        vol_pred = vol_pred.transpose(1, 0, 2, 3)
        vol_true = vol_true.transpose(1, 0, 2, 3)

        for c in range(Config.NUM_CLASSES):
            d = dice_coef(vol_true[c], vol_pred[c])
            h = hausdorff_3d(vol_true[c], vol_pred[c])
            if np.isinf(h):
                h = 1.0  # Cap for metric calculation

            dice_scores_3d.append(d)
            hausdorff_scores_3d.append(h)

    mean_dice = np.mean(dice_scores_3d)
    mean_hausdorff = np.mean(hausdorff_scores_3d)

    # Metric: 0.4 * Dice + 0.6 * Hausdorff_Score
    # Note: Task says Hausdorff is distance. Usually we minimize distance.
    # But "score" implies higher is better.
    # The task description says: "The two metrics are combined, with a weight of 0.4 for the Dice metric and 0.6 for the Hausdorff distance."
    # Usually competitions use (1 - normalized_hausdorff).
    # Based on library.train implementation: combined_score = 0.4 * mean_dice + 0.6 * (1.0 - mean_hausdorff)
    final_metric = 0.4 * mean_dice + 0.6 * (1.0 - mean_hausdorff)

    print(f"Final Validation Metric: {final_metric}")

    # --- 2. Failure Analysis (Correlations) ---
    results_df = pd.DataFrame(results)

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    features = ["slice_idx", "day", "pixel_spacing", "img_width"]
    for feat in features:
        if feat in results_df.columns:
            # Drop NaNs if any
            valid_data = results_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["error"])
                print(f"  Correlation with {feat}: {corr:.4f}")
            else:
                print(f"  Correlation with {feat}: Insufficient data")


def main():
    # 1. Configure for Fast Baseline
    # Override Config defaults to fit time constraints while ensuring functionality
    Config.EPOCHS = 5
    Config.DATA_FRACTION = 0.25  # Use 25% of training data
    Config.DEBUG = (
        False  # We handle subsampling via DATA_FRACTION, not DEBUG mode logic
    )

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Starting Runfile Execution ===")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Data Fraction={Config.DATA_FRACTION}"
    )

    # 2. Train Model
    print("\n--- Phase 1: Training ---")
    # run_training uses the Config settings we just modified
    run_training()

    # 3. Validation & Failure Analysis
    print("\n--- Phase 2: Validation & Failure Analysis ---")

    # Load metadata
    val_df = process_metadata(
        Config.VAL_METADATA_PATH, "val_processed", load_cached_data=True
    )

    # Load best model
    model = GhostUNet(in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print("Loaded best model for analysis.")
    else:
        print("Error: Model file not found after training.")
        return

    model.to(device)

    # Create validation loader (no shuffle, no drop_last)
    val_ds = UWGI25DDataset(val_df, transforms=get_transforms("valid"), mode="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Run analysis
    perform_failure_analysis(model, val_loader, val_df, device)

    # 4. Inference & Submission
    print("\n--- Phase 3: Inference & Submission ---")
    # run_inference handles loading test data, model, and generating submission.csv
    run_inference()

    print("\n=== Execution Complete ===")


if __name__ == "__main__":
    main()
