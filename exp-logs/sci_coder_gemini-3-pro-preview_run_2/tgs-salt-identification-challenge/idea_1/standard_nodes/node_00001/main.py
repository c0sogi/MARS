import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.train import Trainer
from library.predict import generate_submission
from library.dataset import SaltDataset
from library.utils import calculate_competition_metric, set_seed, metric_iou


def main():
    # 1. Configuration
    # We limit epochs to 20 to ensure a fast baseline execution while allowing convergence.
    config = Config(EPOCHS=20, BATCH_SIZE=32)
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    print("Initializing pipeline...")

    # 2. Training
    # Initialize Trainer and start training
    trainer = Trainer(config)
    # We use the full dataset (debug_limit=None) because it is small (2400 images),
    # but we rely on the reduced epoch count for speed.
    trainer.train(epochs=config.EPOCHS)

    # 3. Validation & Metric Calculation
    print("Running validation on hold-out set...")

    # Load validation dataset using cached data for speed
    val_ds = SaltDataset(config.VAL_CSV, config, mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the best model for evaluation
    model = trainer.model
    if os.path.exists(config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))

    model.eval()

    all_preds = []
    all_targets = []

    # Store per-sample stats for failure analysis
    sample_ious = []
    sample_depths = []
    sample_coverages = []

    # Get metadata for failure analysis
    val_df = pd.read_csv(config.VAL_CSV)
    # We assume the loader yields data in the same order as the CSV (shuffle=False)
    # val_ds indices match val_df rows.

    with torch.no_grad():
        for i, (images, depths, masks) in enumerate(val_loader):
            images = images.to(device)
            depths_gpu = depths.to(device)
            masks = masks.to(device)

            # Inference
            outputs = model(images, depths_gpu)
            probs = torch.sigmoid(outputs)

            # Move to CPU for metric calculation to save GPU memory
            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            all_preds.append(probs_np)
            all_targets.append(masks_np)

            # Calculate per-sample IoU (threshold 0.5) for failure analysis
            # We iterate manually to ensure alignment with metadata
            batch_start_idx = i * config.BATCH_SIZE

            for j in range(probs_np.shape[0]):
                p = probs_np[j]
                t = masks_np[j]

                # Simple IoU for failure analysis correlation
                # Flatten
                p_flat = (p > 0.5).astype(np.uint8).flatten()
                t_flat = (t > 0.5).astype(np.uint8).flatten()

                intersection = np.sum(p_flat * t_flat)
                union = np.sum(p_flat) + np.sum(t_flat) - intersection

                if union == 0:
                    iou = 1.0
                else:
                    iou = intersection / union

                sample_ious.append(iou)

                # Retrieve metadata
                global_idx = batch_start_idx + j
                if global_idx < len(val_df):
                    row = val_df.iloc[global_idx]
                    sample_depths.append(row["z"])
                    sample_coverages.append(row["salt_coverage"])

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Calculate Competition Metric (mAP over thresholds)
    final_metric = calculate_competition_metric(y_true, y_pred, threshold=0.5)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("Performing Failure Analysis...")

    # Calculate Error Magnitude (1 - IoU)
    errors = 1.0 - np.array(sample_ious)
    depths = np.array(sample_depths)
    coverages = np.array(sample_coverages)

    # Correlation with Depth
    if len(depths) > 1 and np.std(depths) > 0:
        corr_depth = np.corrcoef(errors, depths)[0, 1]
    else:
        corr_depth = 0.0

    # Correlation with Salt Coverage
    if len(coverages) > 1 and np.std(coverages) > 0:
        corr_coverage = np.corrcoef(errors, coverages)[0, 1]
    else:
        corr_coverage = 0.0

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_coverage:.4f}")

    if abs(corr_depth) > 0.2:
        print("  -> Significant relationship found between depth and model error.")
    if abs(corr_coverage) > 0.2:
        print(
            "  -> Significant relationship found between salt coverage and model error."
        )

    # 5. Submission Generation
    print("Generating submission file...")
    generate_submission(config)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
