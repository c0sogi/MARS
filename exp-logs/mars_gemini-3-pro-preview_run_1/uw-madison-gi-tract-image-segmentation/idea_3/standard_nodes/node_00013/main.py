import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr
from collections import defaultdict

# Import library modules
from library.config import Config
from library.utils import (
    set_seed,
    compute_dice_coefficient,
    compute_hausdorff_distance,
    rle_decode,
    keep_largest_component,
)
from library.data import prepare_loaders
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set by correlating
    model error with input features.
    """
    print("\n=== Running Failure Analysis ===")
    model.eval()

    # 1. Collect Predictions
    preds_map = defaultdict(list)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            slice_infos = batch["slice_info"]

            # Mixed precision inference
            with torch.cuda.amp.autocast():
                outputs = model(images)
                probs = torch.sigmoid(outputs)

            pred_masks = (probs > Config.MASK_THRESHOLD).float().cpu().numpy()

            for i, slice_info in enumerate(slice_infos):
                parts = slice_info.split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                slice_num = int(parts[2])
                preds_map[case_day].append((slice_num, pred_masks[i]))

    # 2. Compute Per-Case Metrics
    df_val = pd.read_csv(Config.VAL_CSV, keep_default_na=False)
    val_groups = df_val.groupby(["case", "day"])

    case_metrics = []

    for (case, day), group in val_groups:
        case_day = f"{case}_{day}"
        if case_day not in preds_map:
            continue

        case_preds = sorted(preds_map[case_day], key=lambda x: x[0])

        h = group.iloc[0]["height"]
        w = group.iloc[0]["width"]
        num_slices = len(group)

        # Initialize GT and Pred Volumes
        gt_vol = np.zeros((num_slices, h, w, Config.NUM_CLASSES), dtype=np.uint8)
        pred_vol = np.zeros((num_slices, h, w, Config.NUM_CLASSES), dtype=np.uint8)

        # Map slice numbers to volume indices
        sorted_group = group.sort_values(
            "slice", key=lambda x: x.astype(int) if x.dtype == "O" else x
        )
        slice_to_idx = {
            int(row.slice): i for i, row in enumerate(sorted_group.itertuples())
        }

        # Fill GT Volume
        for row in sorted_group.itertuples():
            idx = slice_to_idx[int(row.slice)]
            for c_idx, class_name in enumerate(Config.CLASSES):
                rle = getattr(row, class_name)
                if rle:
                    gt_vol[idx, :, :, c_idx] = rle_decode(rle, (h, w))

        # Fill Pred Volume
        for slice_num, mask in case_preds:
            if slice_num in slice_to_idx:
                idx = slice_to_idx[slice_num]
                # Resize mask if needed (model output vs original size)
                if mask.shape[1:] != (h, w):
                    mask_t = mask.transpose(1, 2, 0)
                    mask_resized = cv2.resize(
                        mask_t, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                    if mask_resized.ndim == 2:
                        mask_resized = np.expand_dims(mask_resized, axis=-1)
                    pred_vol[idx] = mask_resized
                else:
                    pred_vol[idx] = mask.transpose(1, 2, 0)

        # Post-process: Keep largest component
        for c in range(Config.NUM_CLASSES):
            pred_vol[..., c] = keep_largest_component(pred_vol[..., c])

        # Compute Metrics
        dice = compute_dice_coefficient(gt_vol, pred_vol)

        hd_scores = []
        for c in range(Config.NUM_CLASSES):
            hd_scores.append(
                compute_hausdorff_distance(gt_vol[..., c], pred_vol[..., c])
            )
        mean_hd = np.mean(hd_scores)

        # Calculate Score and Error
        score = 0.4 * dice + 0.6 * (1.0 - mean_hd)
        error = 1.0 - score

        # Extract Features for Correlation
        gt_pixels = np.sum(gt_vol)
        spacing = group.iloc[0]["spacing_x"]

        case_metrics.append(
            {
                "case_day": case_day,
                "error": error,
                "score": score,
                "num_slices": num_slices,
                "organ_pixels": gt_pixels,
                "pixel_spacing": spacing,
            }
        )

    metrics_df = pd.DataFrame(case_metrics)

    # 3. Correlation Analysis
    print(f"Analyzed {len(metrics_df)} validation cases.")

    features = ["num_slices", "organ_pixels", "pixel_spacing"]
    print("Correlation with Error (1 - Score):")
    for feat in features:
        if len(metrics_df) > 1 and metrics_df[feat].std() > 0:
            corr, _ = pearsonr(metrics_df["error"], metrics_df[feat])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: NaN (Insufficient variance or data)")


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)

    # 2. Data Loading
    print("Preparing data loaders...")
    # We use debug=False to ensure we validate on the full hold-out set
    train_loader, val_loader, test_loader = prepare_loaders(debug=False)

    # 3. Training
    print("Initializing trainer...")
    trainer = Trainer(train_loader, val_loader, test_loader)

    trainer.fit()

    # 4. Final Metric Reporting
    final_metric = trainer.best_score
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # Load best model weights for analysis
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=Config.DEVICE)
        )

    run_failure_analysis(trainer.model, val_loader, Config.DEVICE)

    # 6. Submission Generation
    THRESHOLD = 0.4358463585
    if final_metric > THRESHOLD:
        trainer.predict_submission()
    else:
        print(
            f"Score {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
