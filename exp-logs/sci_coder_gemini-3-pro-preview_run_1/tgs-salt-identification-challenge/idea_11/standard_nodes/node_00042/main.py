import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from library.config import Config
from library.train import train_model, center_crop
from library.inference import predict_ensemble
from library.dataset import SaltDataset
from library.model import HighCapacityUNet
from library.utils import calc_iou, calc_map


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration for Fast Baseline
    # -------------------------------------------------------------------------
    # Override Config defaults to ensure execution finishes quickly (Fast Baseline)
    # while maintaining enough training signal to potentially pass the threshold.
    # A100 GPU can handle 2400 images * 20 epochs very quickly (< 10 mins).
    Config.SEED = 42
    Config.EPOCHS_PER_CYCLE = 20  # Reduced from 50
    Config.NUM_CYCLES = 1  # Reduced from 3
    Config.TOTAL_EPOCHS = Config.EPOCHS_PER_CYCLE * Config.NUM_CYCLES
    Config.DEBUG_DATA_LIMIT = None  # Use full dataset (small enough)

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print(f"Starting Training (Epochs: {Config.TOTAL_EPOCHS})...")
    train_model()

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nStarting Validation & Failure Analysis...")
    device = torch.device(Config.DEVICE)

    # Load the best model saved during training
    model = HighCapacityUNet().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load Validation Data
    # We use the library's dataset class to ensure consistent preprocessing
    val_dataset = SaltDataset(mode="val", load_cached_data=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Metadata for Failure Analysis (Coverage info)
    df_val_meta = pd.read_csv(Config.VAL_CSV)
    # Create a mapping from ID to Coverage
    id_to_coverage = dict(zip(df_val_meta.id, df_val_meta.coverage))

    all_ious = []
    all_depths = []
    all_coverages = []

    # Lists for mAP calculation
    all_preds_np = []
    all_masks_np = []

    with torch.no_grad():
        for images, masks, depths, ids in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            depths_gpu = depths.to(device)

            # Inference
            logits = model(images, depths_gpu)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Crop to original size (101x101) for accurate metric calculation
            preds_cropped = center_crop(preds, Config.ORIG_SIZE, Config.ORIG_SIZE)
            masks_cropped = center_crop(masks, Config.ORIG_SIZE, Config.ORIG_SIZE)

            # Process batch for metrics and analysis
            for i in range(len(ids)):
                # Convert to numpy
                p = preds_cropped[i].cpu().numpy().squeeze()
                t = masks_cropped[i].cpu().numpy().squeeze()

                # Store for mAP
                all_preds_np.append(p)
                all_masks_np.append(t)

                # Calculate IoU for Failure Analysis
                iou = calc_iou(p, t)
                all_ious.append(iou)

                # Store Metadata
                all_depths.append(depths[i].item())
                all_coverages.append(id_to_coverage.get(ids[i], 0.0))

    # Calculate Final Validation Metric (mAP)
    final_metric = calc_map(all_preds_np, all_masks_np)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error (1 - IoU) and Features
    errors = 1.0 - np.array(all_ious)
    depths_arr = np.array(all_depths)
    cov_arr = np.array(all_coverages)

    # Avoid correlation on constant arrays
    if np.std(errors) > 1e-6 and np.std(depths_arr) > 1e-6:
        corr_depth, _ = pearsonr(errors, depths_arr)
    else:
        corr_depth = 0.0

    if np.std(errors) > 1e-6 and np.std(cov_arr) > 1e-6:
        corr_cov, _ = pearsonr(errors, cov_arr)
    else:
        corr_cov = 0.0

    print("\nFailure Analysis:")
    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    # Generate submission only if metric threshold is met
    THRESHOLD = 0.833
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.4f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        # predict_ensemble handles the test set inference and CSV generation
        # It will fallback to best_model.pth since we only ran 1 cycle
        predict_ensemble(debug=False)
    else:
        print(
            f"\nMetric ({final_metric:.4f}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
