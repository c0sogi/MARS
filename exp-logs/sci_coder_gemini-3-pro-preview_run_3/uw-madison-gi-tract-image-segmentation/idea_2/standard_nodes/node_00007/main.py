import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from scipy.stats import pearsonr

# Add the current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.train import run_training
from library.inference import predict, post_process_3d
from library.model import UNetEfficientNet
from library.data import UWDataset, prepare_data, get_transforms
from library.utils import set_seed, dice_coef, hausdorff_3d, rle_decode


def main():
    # 1. Setup and Configuration Overrides for Optimized Run
    set_seed(Config.SEED)

    # Override Config for optimized training
    Config.DEBUG = False
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 64  # Leverage A100 memory

    print("=== Starting Optimized Run ===")
    print(f"Config: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 2. Run Training
    # This will train the model and save 'best_model.pth' to Config.CHECKPOINT_PATH
    run_training(debug=Config.DEBUG, load_cached_data=True)

    # 3. Full Validation & Metric Computation
    print("\n=== Running Full Validation ===")

    # Load full validation metadata
    df_val = pd.read_csv(Config.VAL_CSV)

    # Prepare 2.5D data (this might use cache or generate it)
    df_val = prepare_data(df_val, load_cached_data=True, split="val")

    # Setup Dataset and Loader
    val_dataset = UWDataset(
        df_val, transforms=get_transforms(data="valid"), mode="valid"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE
        * 2,  # Double batch size for inference (no gradients)
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    device = Config.DEVICE
    model = UNetEfficientNet(
        backbone_name=Config.BACKBONE, pretrained=False, classes=Config.NUM_CLASSES
    )

    if os.path.exists(Config.CHECKPOINT_PATH):
        state_dict = torch.load(Config.CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded best model checkpoint.")
    else:
        print(
            "Error: Checkpoint not found! Using random weights (expect poor performance)."
        )

    model.to(device)
    model.eval()

    # Inference Loop
    # We need to store predictions to group them by case/day for 3D metrics
    # Structure: results[case_day] = { 'slice_nums': [], 'preds': [], 'gts': [], 'ids': [] }
    results = defaultdict(lambda: {"slice_nums": [], "preds": [], "gts": [], "ids": []})

    # Also store slice-level metrics for failure analysis
    slice_metrics = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device, dtype=torch.float32)
            # masks are (B, C, H, W)

            # Predict
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > Config.PRED_THR).float().cpu().numpy()

            # Ground truth to numpy
            gts = masks.cpu().numpy()

            # Get metadata for this batch
            # We need to map batch indices back to dataframe rows
            start_idx = i * val_loader.batch_size
            end_idx = start_idx + images.size(0)
            batch_rows = df_val.iloc[start_idx:end_idx]

            for idx, (pred, gt, (_, row)) in enumerate(
                zip(preds, gts, batch_rows.iterrows())
            ):
                # pred/gt shape: (C, H, W)

                # Calculate slice-level Dice for failure analysis
                # Average dice across classes for this slice
                slice_dice = dice_coef(gt, pred)

                slice_metrics.append(
                    {
                        "id": row["id"],
                        "dice": slice_dice,
                        "error": 1.0 - slice_dice,
                        "height": row["height"],
                        "width": row["width"],
                        "pixel_spacing_h": row["pixel_spacing_h"],
                        "pixel_spacing_w": row["pixel_spacing_w"],
                        "slice_num": int(row["slice"]),
                        "case": row["case"],
                        "day": row["day"],
                    }
                )

                # Store for 3D aggregation
                case_day = f"{row['case']}_{row['day']}"
                results[case_day]["slice_nums"].append(int(row["slice"]))
                results[case_day]["preds"].append(pred)
                results[case_day]["gts"].append(gt)
                results[case_day]["ids"].append(row["id"])

    # 4. Compute 3D Metrics
    dice_scores = []
    hausdorff_scores = []

    print("Computing 3D Metrics...")

    for case_day, data in results.items():
        # Sort by slice number to ensure correct 3D volume construction
        sorted_indices = np.argsort(data["slice_nums"])

        # Stack into (Depth, C, H, W)
        vol_pred = np.stack(data["preds"])[sorted_indices]
        vol_gt = np.stack(data["gts"])[sorted_indices]

        # Transpose to (C, Depth, H, W) for easier class iteration
        vol_pred = vol_pred.transpose(1, 0, 2, 3)
        vol_gt = vol_gt.transpose(1, 0, 2, 3)

        # Calculate metrics per class
        for c in range(Config.NUM_CLASSES):
            # Extract binary volumes for class c
            p = vol_pred[c]
            g = vol_gt[c]

            # Apply 3D Post-Processing (CCA) - Cite solution_lesson_node_00005
            p = post_process_3d(p)

            # Dice (3D is same as flattened 2D sum)
            d = dice_coef(g, p)
            dice_scores.append(d)

            # Hausdorff 3D
            h = hausdorff_3d(g, p)
            hausdorff_scores.append(h)

    # Aggregate
    mean_dice = np.mean(dice_scores)
    mean_hausdorff = np.mean(hausdorff_scores)

    # Task Metric: 0.4 * Dice + 0.6 * Hausdorff
    # Note: Hausdorff in the task description: "normalized by image size to create a bounded 0-1 score".
    # The provided hausdorff_3d function does this normalization.
    # However, Hausdorff is a distance (lower is better), while Dice is similarity (higher is better).
    # The prompt says: "Metric: Mean Dice coefficient and 3D Hausdorff distance."
    # And "The two metrics are combined, with a weight of 0.4 for the Dice metric and 0.6 for the Hausdorff distance."
    # Usually, competitions define a score where higher is better.
    # If Hausdorff is distance, 1 - Hausdorff might be the score component, or the metric is simply a weighted sum.
    # Given the standard Kaggle UW-Madison competition metric, it was 0.4*Dice + 0.6*(1 - Hausdorff).
    # However, the prompt strictly says: "combined... 0.4 for Dice... 0.6 for Hausdorff".
    # And "Hausdorff distance... bounded 0-1 score".
    # If the score is to be maximized, and Hausdorff is a distance, we likely need (1 - H).
    # Let's assume the standard interpretation for a "Score" to be maximized: Score = 0.4*Dice + 0.6*(1 - Hausdorff).
    # If the prompt meant minimized, it would be weird to combine similarity (Dice) with distance.
    # I will use Score = 0.4 * Dice + 0.6 * (1.0 - mean_hausdorff).

    final_score = 0.4 * mean_dice + 0.6 * (1.0 - mean_hausdorff)

    print(f"Mean Dice: {mean_dice:.6f}")
    print(f"Mean Hausdorff: {mean_hausdorff:.6f}")
    print(f"Final Validation Metric: {final_score}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    df_metrics = pd.DataFrame(slice_metrics)

    # Normalize slice position per case
    max_slices = df_metrics.groupby(["case", "day"])["slice_num"].transform("max")
    df_metrics["rel_position"] = df_metrics["slice_num"] / max_slices

    # Calculate correlations
    features = ["height", "width", "rel_position"]
    print("Correlation between Error (1-Dice) and Input Features:")
    for feat in features:
        if feat in df_metrics.columns:
            corr, _ = pearsonr(df_metrics["error"], df_metrics[feat])
            print(f"  {feat}: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 0.23448658295996072
    if final_score > THRESHOLD:
        print(
            f"\nMetric ({final_score:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        # Run inference on test set
        # We use the provided library function which handles test set loading and prediction
        predict(load_cached_data=True, debug=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_score:.6f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
