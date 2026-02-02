import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import (
    set_seed,
    get_dice_coef,
    get_3d_hausdorff,
    keep_largest_connected_component_3d,
)
from library.dataset import prepare_loaders
from library.train import Trainer
from library.inference import predict_and_submit
from library.model import UnetPlusPlus

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Configuration & Setup
    print("Setting up configuration...")
    set_seed(Config.SEED)

    # Optimize for speed/time limit (2 hours max)
    # A100 is fast. 12 Epochs on full data (~24k images) is a safe baseline.
    Config.EPOCHS = 12
    Config.BATCH_SIZE = 48
    Config.NUM_WORKERS = 12

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Training
    print(f"Starting training pipeline (Epochs={Config.EPOCHS})...")
    # Load full dataset (debug=False)
    train_loader, val_loader = prepare_loaders(load_cached_data=True, debug=False)

    trainer = Trainer(train_loader, val_loader)
    trainer.fit(epochs=Config.EPOCHS, patience=3)

    # 3. Validation & Failure Analysis
    print("Loading best model for validation and failure analysis...")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Best model not found, using current weights.")

    trainer.model.eval()

    # We reconstruct the validation loop here to get per-case metrics for failure analysis
    val_df = pd.read_csv(Config.VAL_CSV, keep_default_na=False)

    all_preds = []
    all_masks = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(Config.DEVICE, dtype=torch.float32)
            # Eval mode returns single tensor (not list)
            outputs = trainer.model(images)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_masks.append(masks.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # Group by Case+Day for 3D reconstruction
    groups = val_df.groupby(["case", "day"])

    case_metrics = []

    print("Computing 3D metrics per case...")
    for (case, day), group in groups:
        indices = group.index.to_numpy()
        slice_nums = group["slice"].astype(int).values
        sort_idx = np.argsort(slice_nums)

        # Extract volumes sorted by Z-depth
        vol_preds = all_preds[indices][sort_idx]
        vol_masks = all_masks[indices][sort_idx]

        # Metrics for this case
        c_dice = []
        c_hd = []

        # Calculate features for failure analysis
        total_organ_pixels = np.sum(vol_masks)
        num_slices = len(slice_nums)

        for class_idx in range(Config.NUM_CLASSES):
            p_vol = vol_preds[:, class_idx, :, :]
            t_vol = vol_masks[:, class_idx, :, :]

            p_vol_bin = (p_vol > Config.MASK_THRESHOLD).astype(np.uint8)
            t_vol_bin = (t_vol > 0.5).astype(np.uint8)

            # Post-processing: 3D CCA
            p_vol_processed = keep_largest_connected_component_3d(
                p_vol_bin, min_size=Config.MIN_COMPONENT_SIZE
            )

            c_dice.append(get_dice_coef(t_vol_bin, p_vol_processed))
            c_hd.append(get_3d_hausdorff(t_vol_bin, p_vol_processed))

        mean_dice = np.mean(c_dice)
        mean_hd = np.mean(c_hd)
        # Score: 0.4*Dice + 0.6*(1-HD)
        score = 0.4 * mean_dice + 0.6 * (1.0 - mean_hd)

        case_metrics.append(
            {
                "case": case,
                "day": day,
                "score": score,
                "dice": mean_dice,
                "hd": mean_hd,
                "num_slices": num_slices,
                "organ_pixels": total_organ_pixels,
            }
        )

    # Aggregate Final Metric
    metrics_df = pd.DataFrame(case_metrics)
    final_score = metrics_df["score"].mean()

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_score}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    metrics_df["error"] = 1.0 - metrics_df["score"]

    # Correlation with Number of Slices
    if len(metrics_df) > 1:
        corr_slices, _ = pearsonr(metrics_df["error"], metrics_df["num_slices"])
        print(f"Correlation (Error vs Num Slices): {corr_slices:.4f}")

        # Correlation with Organ Size (Total Pixels)
        corr_size, _ = pearsonr(metrics_df["error"], metrics_df["organ_pixels"])
        print(f"Correlation (Error vs Organ Size): {corr_size:.4f}")
    else:
        print("Insufficient validation cases for correlation analysis.")

    # 5. Submission
    THRESHOLD = 0.5184837797359911
    if final_score > THRESHOLD:
        print(
            f"\nScore ({final_score:.5f}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Free memory before inference
        del trainer, all_preds, all_masks, vol_preds, vol_masks
        torch.cuda.empty_cache()

        predict_and_submit()
    else:
        print(
            f"\nScore ({final_score:.5f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
